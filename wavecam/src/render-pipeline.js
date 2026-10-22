import { FFTProcessor } from './fft-processor.js';

export class RenderPipeline {
    constructor(device, context, textureSize) {
        this.device = device;
        this.context = context;
        this.width = textureSize.width;
        this.height = textureSize.height;

        // Rendering resources
        this.renderPipeline = null;
        this.waveOnlyPipeline = null;
        this.bindGroupLayout = null;
        this.waveOnlyBindGroupLayout = null;
        this.bindGroup = null;
        this.vertexBuffer = null;
        this.sampler = null;
        this.renderParamsBuffer = null;

        // Canvas contexts
        this.waveOnlyCanvas = null;
        this.waveOnlyContext = null;

        // Shaders
        this.vertexShader = null;
        this.fragmentShader = null;
        this.waveOnlyFragmentShader = null;

        // FFT processor for convoluted blending
        this.fftProcessor = null;

        // Buffers for FFT processing
        this.fftBuffers = {
            webcamData: null,
            waveData: null,
            convolutedResult: null
        };
    }

    async initialize() {
        await this.createShaders();
        await this.createVertexBuffer();
        await this.createSampler();
        await this.createRenderParamsBuffer();
        await this.setupWaveOnlyCanvas();
        await this.createRenderPipeline();
        await this.createWaveOnlyPipeline();
        await this.initializeFFTProcessor();

        console.log('Render pipeline initialized');
    }

    async initializeFFTProcessor() {
        this.fftProcessor = new FFTProcessor(this.device, this.width, this.height);
        await this.fftProcessor.initialize();

        // Create buffers for texture data extraction
        const bufferSize = this.width * this.height * 4; // 4 bytes per float

        this.fftBuffers.webcamData = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Webcam data for FFT'
        });

        this.fftBuffers.waveData = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Wave data for FFT'
        });

        this.fftBuffers.convolutedResult = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: 'Convoluted result'
        });

        console.log('FFT processor initialized for convoluted blending');
    }

    async createShaders() {
        // Vertex shader - simple fullscreen quad
        const vertexShaderCode = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}`;

        // Fragment shader - visualize wave simulation with webcam overlay
        const fragmentShaderCode = `
@group(0) @binding(0) var waveTexture: texture_2d<f32>;
@group(0) @binding(1) var webcamTexture: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var<uniform> renderParams: RenderParams;

struct RenderParams {
    mix_ratio: f32,
    color_scheme: f32,
    blend_mode: f32,
    time: f32,
    color_scale_min: f32,
    color_scale_max: f32,
    enable_normal_mapping: f32,
    normal_strength: f32,
    light_direction: f32,
    light_elevation: f32,
    enable_chromatic_aberration: f32,
    aberration_strength: f32,
    aberration_offset: f32,
    aberration_angle: f32,
    _padding1: f32,
    _padding2: f32,
}

// Helper functions for color interpolation
fn mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * clamp(t, 0.0, 1.0);
}

// Surface normal calculation from height field
fn calculateSurfaceNormal(uv: vec2<f32>, heightStrength: f32) -> vec3<f32> {
    let texelSize = 1.0 / vec2<f32>(textureDimensions(waveTexture));
    
    // Sample neighboring heights to calculate gradients
    let heightL = textureSample(waveTexture, textureSampler, uv + vec2<f32>(-texelSize.x, 0.0)).r;
    let heightR = textureSample(waveTexture, textureSampler, uv + vec2<f32>(texelSize.x, 0.0)).r;
    let heightU = textureSample(waveTexture, textureSampler, uv + vec2<f32>(0.0, -texelSize.y)).r;
    let heightD = textureSample(waveTexture, textureSampler, uv + vec2<f32>(0.0, texelSize.y)).r;
    
    // Calculate gradients (partial derivatives)
    let dhdx = (heightR - heightL) * 0.5 * heightStrength;
    let dhdy = (heightD - heightU) * 0.5 * heightStrength;
    
    // Surface normal from height gradients
    // Normal = normalize((-dh/dx, -dh/dy, 1))
    let normal = normalize(vec3<f32>(-dhdx, -dhdy, 1.0));
    
    return normal;
}

// Calculate lighting from surface normal and light direction
fn calculateLighting(normal: vec3<f32>, lightDir: vec3<f32>) -> f32 {
    // Simple Lambertian diffuse lighting
    let dotProduct = dot(normal, lightDir);
    return max(dotProduct, 0.1); // Ambient light minimum
}

// Color scheme functions with 28 different color maps
fn getColorScheme(value: f32, scheme: f32) -> vec3<f32> {
    // Map the value from [colorScaleMin, colorScaleMax] to [0, 1] for color mapping
    let range = renderParams.color_scale_max - renderParams.color_scale_min;
    let normalized_value = (value - renderParams.color_scale_min) / range;
    let clamped_value = clamp(normalized_value, 0.0, 1.0);
    
    // For diverging color schemes (like red-blue), we want to map around 0.5
    // For sequential schemes, we use the full 0-1 range
    let t = clamped_value;
    
    // Classic (0-1)
    if (scheme < 0.5) {
        // Red-Blue (Classic) - diverging color scheme
        // Map t=0 to blue, t=0.5 to white/neutral, t=1 to red
        if (t < 0.5) {
            // Blue to white
            let intensity = t * 2.0;
            return vec3<f32>(intensity, intensity, 1.0);
        } else {
            // White to red
            let intensity = (t - 0.5) * 2.0;
            return vec3<f32>(1.0, 1.0 - intensity, 1.0 - intensity);
        }
    } else if (scheme < 1.5) {
        // Grayscale - use t for proper scaling
        return vec3<f32>(t, t, t);
    }
    
    // Scientific (2-7)
    else if (scheme < 2.5) {
        // Viridis (Purple-Blue-Green-Yellow)
        if (t < 0.25) {
            return mix3(vec3<f32>(0.267, 0.004, 0.329), vec3<f32>(0.282, 0.140, 0.457), t * 4.0);
        } else if (t < 0.5) {
            return mix3(vec3<f32>(0.282, 0.140, 0.457), vec3<f32>(0.253, 0.265, 0.529), (t - 0.25) * 4.0);
        } else if (t < 0.75) {
            return mix3(vec3<f32>(0.253, 0.265, 0.529), vec3<f32>(0.163, 0.471, 0.558), (t - 0.5) * 4.0);
        } else {
            return mix3(vec3<f32>(0.163, 0.471, 0.558), vec3<f32>(0.993, 0.906, 0.144), (t - 0.75) * 4.0);
        }
    } else if (scheme < 3.5) {
        // Plasma
        if (t < 0.33) {
            return mix3(vec3<f32>(0.050, 0.029, 0.527), vec3<f32>(0.302, 0.019, 0.615), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(0.302, 0.019, 0.615), vec3<f32>(0.716, 0.214, 0.475), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(0.716, 0.214, 0.475), vec3<f32>(0.940, 0.975, 0.131), (t - 0.66) * 3.0);
        }
    } else if (scheme < 4.5) {
        // Inferno
        if (t < 0.33) {
            return mix3(vec3<f32>(0.000, 0.000, 0.014), vec3<f32>(0.188, 0.043, 0.415), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(0.188, 0.043, 0.415), vec3<f32>(0.735, 0.215, 0.330), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(0.735, 0.215, 0.330), vec3<f32>(0.988, 0.998, 0.645), (t - 0.66) * 3.0);
        }
    } else if (scheme < 5.5) {
        // Magma
        if (t < 0.33) {
            return mix3(vec3<f32>(0.000, 0.000, 0.016), vec3<f32>(0.216, 0.063, 0.394), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(0.216, 0.063, 0.394), vec3<f32>(0.698, 0.258, 0.435), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(0.698, 0.258, 0.435), vec3<f32>(0.987, 0.991, 0.749), (t - 0.66) * 3.0);
        }
    } else if (scheme < 6.5) {
        // Cividis
        if (t < 0.5) {
            return mix3(vec3<f32>(0.000, 0.135, 0.304), vec3<f32>(0.329, 0.353, 0.490), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.329, 0.353, 0.490), vec3<f32>(1.000, 0.832, 0.000), (t - 0.5) * 2.0);
        }
    } else if (scheme < 7.5) {
        // Turbo
        if (t < 0.25) {
            return mix3(vec3<f32>(0.190, 0.070, 0.490), vec3<f32>(0.000, 0.550, 0.800), t * 4.0);
        } else if (t < 0.5) {
            return mix3(vec3<f32>(0.000, 0.550, 0.800), vec3<f32>(0.000, 0.850, 0.550), (t - 0.25) * 4.0);
        } else if (t < 0.75) {
            return mix3(vec3<f32>(0.000, 0.850, 0.550), vec3<f32>(0.950, 0.850, 0.000), (t - 0.5) * 4.0);
        } else {
            return mix3(vec3<f32>(0.950, 0.850, 0.000), vec3<f32>(0.480, 0.000, 0.000), (t - 0.75) * 4.0);
        }
    }
    
    // Natural (8-13)
    else if (scheme < 8.5) {
        // Fire
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.5) * 2.0);
        }
    } else if (scheme < 9.5) {
        // Ocean
        return mix3(vec3<f32>(0.0, 0.0, 0.2), vec3<f32>(0.0, 0.8, 1.0), t);
    } else if (scheme < 10.5) {
        // Earth
        if (t < 0.33) {
            return mix3(vec3<f32>(0.2, 0.1, 0.0), vec3<f32>(0.4, 0.3, 0.1), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(0.4, 0.3, 0.1), vec3<f32>(0.2, 0.6, 0.2), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(0.2, 0.6, 0.2), vec3<f32>(0.0, 0.4, 0.8), (t - 0.66) * 3.0);
        }
    } else if (scheme < 11.5) {
        // Sunset
        if (t < 0.5) {
            return mix3(vec3<f32>(0.3, 0.0, 0.5), vec3<f32>(1.0, 0.3, 0.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(1.0, 0.3, 0.0), vec3<f32>(1.0, 1.0, 0.3), (t - 0.5) * 2.0);
        }
    } else if (scheme < 12.5) {
        // Forest
        return mix3(vec3<f32>(0.0, 0.2, 0.0), vec3<f32>(0.5, 1.0, 0.3), t);
    } else if (scheme < 13.5) {
        // Ice
        return mix3(vec3<f32>(0.9, 0.9, 1.0), vec3<f32>(0.0, 0.5, 1.0), t);
    }
    
    // Artistic (14-19)
    else if (scheme < 14.5) {
        // Rainbow
        let hue = t * 6.0;
        let c = 1.0;
        let x = c * (1.0 - abs((hue % 2.0) - 1.0));
        if (hue < 1.0) { return vec3<f32>(c, x, 0.0); }
        else if (hue < 2.0) { return vec3<f32>(x, c, 0.0); }
        else if (hue < 3.0) { return vec3<f32>(0.0, c, x); }
        else if (hue < 4.0) { return vec3<f32>(0.0, x, c); }
        else if (hue < 5.0) { return vec3<f32>(x, 0.0, c); }
        else { return vec3<f32>(c, 0.0, x); }
    } else if (scheme < 15.5) {
        // Neon
        if (t < 0.33) {
            return mix3(vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.66) * 3.0);
        }
    } else if (scheme < 16.5) {
        // Pastel
        if (t < 0.5) {
            return mix3(vec3<f32>(0.9, 0.7, 0.9), vec3<f32>(0.7, 0.9, 0.9), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.7, 0.9, 0.9), vec3<f32>(0.9, 0.9, 0.7), (t - 0.5) * 2.0);
        }
    } else if (scheme < 17.5) {
        // Cyberpunk
        if (t < 0.5) {
            return mix3(vec3<f32>(1.0, 0.0, 0.5), vec3<f32>(0.0, 1.0, 1.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.5, 0.0, 1.0), (t - 0.5) * 2.0);
        }
    } else if (scheme < 18.5) {
        // Vintage
        return mix3(vec3<f32>(0.4, 0.3, 0.2), vec3<f32>(1.0, 0.8, 0.4), t);
    } else if (scheme < 19.5) {
        // Aurora
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 1.0, 0.3), vec3<f32>(0.5, 0.0, 1.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.5, 0.0, 1.0), vec3<f32>(0.0, 0.3, 1.0), (t - 0.5) * 2.0);
        }
    }
    
    // Thermal (20-23)
    else if (scheme < 20.5) {
        // Thermal
        if (t < 0.33) {
            return mix3(vec3<f32>(0.5, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.5, 0.0), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.66) * 3.0);
        }
    } else if (scheme < 21.5) {
        // Hot
        if (t < 0.33) {
            return mix3(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), t * 3.0);
        } else if (t < 0.66) {
            return mix3(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (t - 0.33) * 3.0);
        } else {
            return mix3(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), (t - 0.66) * 3.0);
        }
    } else if (scheme < 22.5) {
        // Cool
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), (t - 0.5) * 2.0);
        }
    } else if (scheme < 23.5) {
        // Copper
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.8, 0.3, 0.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.8, 0.3, 0.0), vec3<f32>(1.0, 0.8, 0.2), (t - 0.5) * 2.0);
        }
    }
    
    // Diverging (24-27)
    else if (scheme < 24.5) {
        // Spectral
        if (t < 0.5) {
            return mix3(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), (t - 0.5) * 2.0);
        }
    } else if (scheme < 25.5) {
        // Cool-Warm
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), t * 2.0);
        } else {
            return mix3(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
        }
    } else if (scheme < 26.5) {
        // Seismic
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 0.0, 0.8), vec3<f32>(0.9, 0.9, 0.9), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.9, 0.9, 0.9), vec3<f32>(0.8, 0.0, 0.0), (t - 0.5) * 2.0);
        }
    } else {
        // Balance
        if (t < 0.5) {
            return mix3(vec3<f32>(0.0, 0.2, 0.6), vec3<f32>(0.5, 0.5, 0.5), t * 2.0);
        } else {
            return mix3(vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.8, 0.4, 0.0), (t - 0.5) * 2.0);
        }
    }
}

// Blending mode functions
fn blendNormal(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(base, overlay, alpha);
}

fn blendAdd(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    return base + overlay * alpha;
}

fn blendSubtract(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    return base - overlay * alpha;
}

fn blendMultiply(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(base, base * overlay, alpha);
}

fn blendScreen(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    let screen = 1.0 - (1.0 - base) * (1.0 - overlay);
    return mix(base, screen, alpha);
}

fn blendOverlay(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    let overlay_result = select(
        2.0 * base * overlay,
        1.0 - 2.0 * (1.0 - base) * (1.0 - overlay),
        base > vec3<f32>(0.5)
    );
    return mix(base, overlay_result, alpha);
}

fn blendDifference(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(base, abs(base - overlay), alpha);
}

fn blendExclusion(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    let exclusion = base + overlay - 2.0 * base * overlay;
    return mix(base, exclusion, alpha);
}

fn blendConvoluted(base: vec3<f32>, overlay: vec3<f32>, alpha: f32) -> vec3<f32> {
    // Experimental convoluted blending - creates a complex interaction pattern
    // This is a visual placeholder that demonstrates the concept while FFT processing occurs
    let time_factor = sin(renderParams.time * 0.5) * 0.5 + 0.5;
    
    // Create frequency-like mixing patterns
    let freq_mix = base * overlay * 2.0;
    let phase_shift = overlay * cos(renderParams.time) + base * sin(renderParams.time * 1.3);
    let convolved = mix(freq_mix, phase_shift, time_factor);
    
    // Add some spectral-like coloring to indicate this is frequency domain processing
    let spectral_tint = vec3<f32>(
        0.8 + 0.2 * sin(renderParams.time),
        0.9 + 0.1 * cos(renderParams.time * 1.2), 
        1.0
    );
    
    let final_result = convolved * spectral_tint;
    return mix(base, final_result, alpha);
}

fn applyBlendMode(base: vec3<f32>, overlay: vec3<f32>, mode: f32, alpha: f32) -> vec3<f32> {
    if (mode < 0.5) {
        return blendNormal(base, overlay, alpha);
    } else if (mode < 1.5) {
        return blendAdd(base, overlay, alpha);
    } else if (mode < 2.5) {
        return blendSubtract(base, overlay, alpha);
    } else if (mode < 3.5) {
        return blendMultiply(base, overlay, alpha);
    } else if (mode < 4.5) {
        return blendScreen(base, overlay, alpha);
    } else if (mode < 5.5) {
        return blendOverlay(base, overlay, alpha);
    } else if (mode < 6.5) {
        return blendDifference(base, overlay, alpha);
    } else if (mode < 7.5) {
        return blendExclusion(base, overlay, alpha);
    } else {
        return blendConvoluted(base, overlay, alpha);
    }
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Sample wave simulation data - ALWAYS in uniform control flow
    let waveSample = textureSample(waveTexture, textureSampler, uv);
    let waveAmplitude = waveSample.r;
    
    // Sample webcam
    let webcamColor = textureSample(webcamTexture, textureSampler, uv).rgb;
    
    // Calculate surface normal if needed (in uniform control flow)
    var surfaceNormal = vec3<f32>(0.0, 0.0, 1.0);
    if (renderParams.enable_normal_mapping > 0.5) {
        surfaceNormal = calculateSurfaceNormal(uv, renderParams.normal_strength);
    }
    
    // Check if this is pre-colored data using alpha channel as flag
    // Alpha = 0.5 indicates pre-colored RGB data (from Schrödinger Mode 3)
    // Alpha = 1.0 indicates scalar data that needs color mapping
    // This is more robust than checking RGB differences
    let isPrecolored = waveSample.a < 0.75;
    
    var waveColor: vec3<f32>;
    
    // Branch on pre-colored vs scalar (OK for non-texture operations)
    if (isPrecolored) {
        // Pre-colored mode: use RGB directly
        waveColor = waveSample.rgb;
    } else {
        // Scalar mode: apply color scheme
        if (renderParams.enable_normal_mapping > 0.5) {
            // Convert light direction and elevation to 3D vector
            let lightDirRad = renderParams.light_direction * 3.14159 / 180.0;
            let lightElevRad = renderParams.light_elevation * 3.14159 / 180.0;
            
            let lightDir = vec3<f32>(
                cos(lightElevRad) * cos(lightDirRad),
                cos(lightElevRad) * sin(lightDirRad),
                sin(lightElevRad)
            );
            
            // Calculate lighting intensity
            let lighting = calculateLighting(surfaceNormal, lightDir);
            
            // Apply color scheme to the base wave data
            let baseColor = getColorScheme(waveAmplitude, renderParams.color_scheme);
            
            // Modulate color by lighting
            waveColor = baseColor * lighting;
            
            // Optional: Add some specular highlights
            let viewDir = vec3<f32>(0.0, 0.0, 1.0); // Camera looking down
            let reflectDir = reflect(-lightDir, surfaceNormal);
            let specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.3;
            waveColor += vec3<f32>(specular);
        } else {
            // Standard color scheme application
            waveColor = getColorScheme(waveAmplitude, renderParams.color_scheme);
        }
    }
    
    // Apply blending mode to combine wave and webcam
    let finalColor = applyBlendMode(
        webcamColor,                     // Base layer (webcam)
        waveColor,                       // Overlay layer (wave visualization)
        renderParams.blend_mode,         // Blending mode
        renderParams.mix_ratio           // Alpha/intensity
    );
    
    // Apply chromatic aberration effect if enabled
    var outputColor = finalColor;
    if (renderParams.enable_chromatic_aberration > 0.5) {
        outputColor = applyChromaticAberration(uv, finalColor);
    }
    
    return vec4<f32>(outputColor, 1.0);
}

// Chromatic aberration effect function
fn applyChromaticAberration(uv: vec2<f32>, originalColor: vec3<f32>) -> vec3<f32> {
    // Convert angle to radians
    let angleRad = renderParams.aberration_angle * 3.14159 / 180.0;
    
    // Calculate offset direction
    let offsetDir = vec2<f32>(cos(angleRad), sin(angleRad));
    
    // Get texture dimensions for pixel-perfect offsets
    let texSize = vec2<f32>(textureDimensions(waveTexture));
    let pixelSize = 1.0 / texSize;
    
    // Calculate base offset in UV space
    let baseOffset = renderParams.aberration_offset * pixelSize;
    
    // Create distance-based falloff from center for more realistic aberration
    let centerDist = length(uv - vec2<f32>(0.5, 0.5)) * 2.0; // 0 at center, 1 at corners
    let distanceFactor = centerDist * centerDist; // Quadratic falloff
    
    // Scale offset by strength and distance
    let scaledOffset = baseOffset * renderParams.aberration_strength * (1.0 + distanceFactor * 2.0);
    
    // Calculate channel offsets (classic 80s style: red and blue separated, green in middle)
    let redOffset = clamp(uv + offsetDir * scaledOffset, vec2<f32>(0.0), vec2<f32>(1.0));
    let blueOffset = clamp(uv - offsetDir * scaledOffset, vec2<f32>(0.0), vec2<f32>(1.0));
    
    // Always sample textures (uniform control flow) - clamp ensures valid coordinates
    let redWave = textureSample(waveTexture, textureSampler, redOffset).r;
    let redWebcam = textureSample(webcamTexture, textureSampler, redOffset).rgb;
    let blueWave = textureSample(waveTexture, textureSampler, blueOffset).r;
    let blueWebcam = textureSample(webcamTexture, textureSampler, blueOffset).rgb;
    
    // Process red channel
    var redWaveColor: vec3<f32>;
    if (renderParams.enable_normal_mapping > 0.5) {
        let redSurfaceNormal = calculateSurfaceNormal(redOffset, renderParams.normal_strength);
        let lightDirRad = renderParams.light_direction * 3.14159 / 180.0;
        let lightElevRad = renderParams.light_elevation * 3.14159 / 180.0;
        let lightDir = vec3<f32>(
            cos(lightElevRad) * cos(lightDirRad),
            cos(lightElevRad) * sin(lightDirRad),
            sin(lightElevRad)
        );
        let lighting = calculateLighting(redSurfaceNormal, lightDir);
        let baseColor = getColorScheme(redWave, renderParams.color_scheme);
        redWaveColor = baseColor * lighting;
        let viewDir = vec3<f32>(0.0, 0.0, 1.0);
        let reflectDir = reflect(-lightDir, redSurfaceNormal);
        let specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.3;
        redWaveColor += vec3<f32>(specular);
    } else {
        redWaveColor = getColorScheme(redWave, renderParams.color_scheme);
    }
    let redSample = applyBlendMode(redWebcam, redWaveColor, renderParams.blend_mode, renderParams.mix_ratio);
    
    // Process blue channel
    var blueWaveColor: vec3<f32>;
    if (renderParams.enable_normal_mapping > 0.5) {
        let blueSurfaceNormal = calculateSurfaceNormal(blueOffset, renderParams.normal_strength);
        let lightDirRad = renderParams.light_direction * 3.14159 / 180.0;
        let lightElevRad = renderParams.light_elevation * 3.14159 / 180.0;
        let lightDir = vec3<f32>(
            cos(lightElevRad) * cos(lightDirRad),
            cos(lightElevRad) * sin(lightDirRad),
            sin(lightElevRad)
        );
        let lighting = calculateLighting(blueSurfaceNormal, lightDir);
        let baseColor = getColorScheme(blueWave, renderParams.color_scheme);
        blueWaveColor = baseColor * lighting;
        let viewDir = vec3<f32>(0.0, 0.0, 1.0);
        let reflectDir = reflect(-lightDir, blueSurfaceNormal);
        let specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.3;
        blueWaveColor += vec3<f32>(specular);
    } else {
        blueWaveColor = getColorScheme(blueWave, renderParams.color_scheme);
    }
    let blueSample = applyBlendMode(blueWebcam, blueWaveColor, renderParams.blend_mode, renderParams.mix_ratio);
    
    // Combine channels with chromatic separation
    let aberratedColor = vec3<f32>(
        redSample.r,      // Red channel from offset position
        originalColor.g,  // Green channel from original position  
        blueSample.b      // Blue channel from opposite offset position
    );
    
    // Add some subtle color fringing for authentic 80s look
    let fringing = sin(uv.x * texSize.x * 0.5) * 0.02;
    let fringedColor = aberratedColor + vec3<f32>(fringing, -fringing * 0.5, -fringing);
    
    return clamp(fringedColor, vec3<f32>(0.0), vec3<f32>(1.0));
}`;

        this.vertexShader = this.device.createShaderModule({
            code: vertexShaderCode
        });

        this.fragmentShader = this.device.createShaderModule({
            code: fragmentShaderCode
        });

        // Wave-only fragment shader for debugging
        const waveOnlyFragmentShaderCode = `
@group(0) @binding(0) var waveTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Sample wave simulation data directly
    let waveColor = textureSample(waveTexture, textureSampler, uv).rgb;
    
    // Enhance the wave visualization for debugging
    let waveIntensity = length(waveColor);
    let enhancedWave = waveColor * (2.0 + waveIntensity);
    
    // Add grid lines to verify we're actually rendering
    let gridX = step(0.98, fract(uv.x * 20.0));
    let gridY = step(0.98, fract(uv.y * 20.0));
    let grid = max(gridX, gridY) * 0.2;
    
    let result = enhancedWave + vec3<f32>(grid);
    
    return vec4<f32>(result, 1.0);
}`;

        this.waveOnlyFragmentShader = this.device.createShaderModule({
            code: waveOnlyFragmentShaderCode
        });
    }

    async createVertexBuffer() {
        // We're using vertex_index in the shader, so we don't need an actual vertex buffer
        // But we'll create a dummy one for compatibility
        this.vertexBuffer = this.device.createBuffer({
            size: 6 * 2 * 4, // 6 vertices, 2 components, 4 bytes each
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true
        });

        // Fill with dummy data since we're generating vertices in the shader
        const vertices = new Float32Array(this.vertexBuffer.getMappedRange());
        vertices.fill(0);
        this.vertexBuffer.unmap();
    }

    async createSampler() {
        this.sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge'
        });
    }

    async createRenderParamsBuffer() {
        this.renderParamsBuffer = this.device.createBuffer({
            size: 64, // 16 f32 values (including chromatic aberration params and padding)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Render parameters buffer'
        });
    }

    async setupWaveOnlyCanvas() {
        // Get the wave-only canvas
        this.waveOnlyCanvas = document.getElementById('wave-only-canvas');
        if (!this.waveOnlyCanvas) {
            throw new Error('Wave-only canvas element not found');
        }

        // Get WebGPU context for the wave-only canvas
        this.waveOnlyContext = this.waveOnlyCanvas.getContext('webgpu');
        if (!this.waveOnlyContext) {
            throw new Error('Failed to get WebGPU context for wave-only canvas');
        }

        // Configure the wave-only canvas
        this.waveOnlyContext.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied'
        });
    }

    async createRenderPipeline() {
        // Create bind group layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Create render pipeline
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.bindGroupLayout]
            }),
            vertex: {
                module: this.vertexShader,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.fragmentShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });
    }

    async createWaveOnlyPipeline() {
        // Create bind group layout for wave-only rendering (only needs wave texture and sampler)
        this.waveOnlyBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}
                }
            ]
        });

        // Create wave-only render pipeline
        this.waveOnlyPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.waveOnlyBindGroupLayout]
            }),
            vertex: {
                module: this.vertexShader,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.waveOnlyFragmentShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });
    }

    createBindGroup(waveTexture, webcamTexture, renderParams) {
        return this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: waveTexture.createView()
                },
                {
                    binding: 1,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 2,
                    resource: this.sampler
                },
                {
                    binding: 3,
                    resource: { buffer: this.renderParamsBuffer }
                }
            ]
        });
    }

    updateRenderParams(mixRatio, colorScheme, blendMode, time, colorScaleMin = -1.0, colorScaleMax = 1.0, enableNormalMapping = false, normalStrength = 1.0, lightDirection = 45.0, lightElevation = 30.0, enableChromaticAberration = false, aberrationStrength = 2.0, aberrationOffset = 0.5, aberrationAngle = 0.0) {
        // Map color scheme names to numbers
        const colorSchemeMap = {
            // Classic
            'red-blue': 0.0,
            'grayscale': 1.0,

            // Scientific
            'viridis': 2.0,
            'plasma': 3.0,
            'inferno': 4.0,
            'magma': 5.0,
            'cividis': 6.0,
            'turbo': 7.0,

            // Natural
            'fire': 8.0,
            'ocean': 9.0,
            'earth': 10.0,
            'sunset': 11.0,
            'forest': 12.0,
            'ice': 13.0,

            // Artistic
            'rainbow': 14.0,
            'neon': 15.0,
            'pastel': 16.0,
            'cyberpunk': 17.0,
            'vintage': 18.0,
            'aurora': 19.0,

            // Thermal
            'thermal': 20.0,
            'hot': 21.0,
            'cool': 22.0,
            'copper': 23.0,

            // Diverging
            'spectral': 24.0,
            'coolwarm': 25.0,
            'seismic': 26.0,
            'balance': 27.0
        };

        // Map blend mode names to numbers
        const blendModeMap = {
            'normal': 0.0,
            'add': 1.0,
            'subtract': 2.0,
            'multiply': 3.0,
            'screen': 4.0,
            'overlay': 5.0,
            'difference': 6.0,
            'exclusion': 7.0,
            'convoluted': 8.0
        };

        const renderParams = new Float32Array([
            mixRatio,
            colorSchemeMap[colorScheme] || 0.0,
            blendModeMap[blendMode] || 0.0,
            time,
            colorScaleMin,
            colorScaleMax,
            enableNormalMapping ? 1.0 : 0.0,
            normalStrength,
            lightDirection,
            lightElevation,
            enableChromaticAberration ? 1.0 : 0.0,
            aberrationStrength,
            aberrationOffset,
            aberrationAngle,
            0.0, // padding
            0.0  // padding
        ]);

        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParams);
    }

    createWaveOnlyBindGroup(waveTexture) {
        return this.device.createBindGroup({
            layout: this.waveOnlyBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: waveTexture.createView()
                },
                {
                    binding: 1,
                    resource: this.sampler
                }
            ]
        });
    }

    render(waveTexture, webcamTexture, renderParams = {}) {
        if (!waveTexture || !webcamTexture) {
            return;
        }

        // Check if convoluted blending is enabled
        const isConvolutedBlending = renderParams.blendMode === 'convoluted';

        // If convoluted blending is enabled, perform FFT processing
        if (isConvolutedBlending && this.fftProcessor) {
            this.performConvolutedBlending(waveTexture, webcamTexture);
        }

        // Update render parameters
        this.updateRenderParams(
            renderParams.mixRatio || 0.7,
            renderParams.colorScheme || 'red-blue',
            renderParams.blendMode || 'normal',
            renderParams.time || 0.0,
            renderParams.colorScaleMin || -1.0,
            renderParams.colorScaleMax || 1.0,
            renderParams.enableNormalMapping || false,
            renderParams.normalStrength || 1.0,
            renderParams.lightDirection || 45.0,
            renderParams.lightElevation || 30.0,
            renderParams.enableChromaticAberration || false,
            renderParams.aberrationStrength || 2.0,
            renderParams.aberrationOffset || 0.5,
            renderParams.aberrationAngle || 0.0
        );

        // Create bind group for this frame
        const bindGroup = this.createBindGroup(waveTexture, webcamTexture);

        // Get current canvas texture
        const canvasTexture = this.context.getCurrentTexture();

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();

        // Create render pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        // Set pipeline and bind group
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, bindGroup);

        // Draw fullscreen quad
        renderPass.draw(6);

        renderPass.end();

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Also render wave-only view
        this.renderWaveOnly(waveTexture);
    }

    async performConvolutedBlending(waveTexture, webcamTexture) {
        try {
            // Extract texture data to buffers
            await this.extractTextureToBuffer(webcamTexture, this.fftBuffers.webcamData);
            await this.extractTextureToBuffer(waveTexture, this.fftBuffers.waveData);

            // Perform FFT convolution
            await this.fftProcessor.performConvolutedBlending(
                this.fftBuffers.webcamData,
                this.fftBuffers.waveData,
                this.fftBuffers.convolutedResult
            );

            console.log('Convoluted blending performed via FFT');
        } catch (error) {
            console.error('Error in convoluted blending:', error);
        }
    }

    async extractTextureToBuffer(texture, buffer) {
        // Create a compute shader to extract texture data to buffer
        if (!this.textureToBufferPipeline) {
            await this.createTextureToBufferPipeline();
        }

        const bindGroup = this.device.createBindGroup({
            layout: this.textureToBufferPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: texture.createView() },
                { binding: 1, resource: { buffer: buffer } }
            ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.textureToBufferPipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.width / 8),
            Math.ceil(this.height / 8)
        );
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    async createTextureToBufferPipeline() {
        const shaderCode = `
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let dims = textureDimensions(input_texture);
    
    if (x >= dims.x || y >= dims.y) {
        return;
    }
    
    let idx = y * dims.x + x;
    let color = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
    
    // Convert to grayscale for FFT processing
    let grayscale = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    output_buffer[idx] = grayscale;
}`;

        const shaderModule = this.device.createShaderModule({ code: shaderCode });

        this.textureToBufferPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });
    }

    renderWaveOnly(waveTexture) {
        if (!waveTexture) {
            return;
        }

        // Create bind group for wave-only rendering
        const waveOnlyBindGroup = this.createWaveOnlyBindGroup(waveTexture);

        // Get current wave-only canvas texture
        const waveOnlyCanvasTexture = this.waveOnlyContext.getCurrentTexture();

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();

        // Create render pass for wave-only canvas
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: waveOnlyCanvasTexture.createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.1, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        // Set pipeline and bind group
        renderPass.setPipeline(this.waveOnlyPipeline);
        renderPass.setBindGroup(0, waveOnlyBindGroup);

        // Draw fullscreen quad
        renderPass.draw(6);

        renderPass.end();

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    renderWebcamOnly(webcamTexture) {
        // Simple passthrough rendering of webcam
        const fragmentShaderCode = `
@group(0) @binding(0) var webcamTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(webcamTexture, textureSampler, uv);
}`;

        // Implementation would require creating a separate pipeline
        // For now, this is a placeholder for future enhancement
    }

    cleanup() {
        if (this.vertexBuffer) {
            this.vertexBuffer.destroy();
        }
    }
}
