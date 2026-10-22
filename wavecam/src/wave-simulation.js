export class WaveSimulation {
    constructor(device, textureSize, shaderManager) {
        this.device = device;
        this.width = textureSize.width;
        this.height = textureSize.height;
        this.shaderManager = shaderManager;

        // Simulation state - proper triple buffering for wave equation
        this.bufferIndices = {
            prev: 0,  // u(t-1)
            curr: 1,  // u(t)
            next: 2   // u(t+1) - output
        };
        this.time = 0.0;
        this.dt = 0.02; // Balanced timestep for stability and speed
        this.dx = 1.5; // Spatial step balanced for wave propagation

        // WebGPU resources
        this.buffers = {
            waveBuffers: [null, null, null], // Triple buffer for u_prev, u_curr, u_next
            boundary: null,
            params: null,
            webcamPrev: null,  // Previous webcam frame for motion detection
            webcamCurr: null,  // Current webcam frame for motion detection

            // Fluid simulation buffers
            velocityX: [null, null], // Double buffer for velocity X component
            velocityY: [null, null], // Double buffer for velocity Y component
            density: [null, null],   // Double buffer for density/dye field
            pressure: null,          // Pressure field for incompressibility
            divergence: null,        // Velocity divergence field

            // Schrödinger equation buffers
            psiR: [null, null],      // Double buffer for real part ψ_R
            psiI: [null, null],      // Double buffer for imaginary part ψ_I
            potential: null          // V(x,y) potential from webcam
        };

        // Schrödinger buffer index for double buffering
        this.schrodingerBufferIndex = 0;

        this.textures = {
            output: null,
            outputView: null
        };

        this.computePipeline = null;
        this.bindGroups = [];
        this.bindGroupLayout = null;

        // Simulation parameters
        this.params = {
            width: this.width,
            height: this.height,
            dt: this.dt,
            dx: this.dx,
            waveSpeed: 2.0,
            damping: 0.0,
            sourceFrequency: 0.1,
            sourceEnabled: 1.0,
            time: 0.0,
            boundaryThreshold: 0.25,
            sourceStrength: 2.0,
            sourceSize: 5.0,
            normalizeWaves: true,
            waveRange: 1.0,
            stepsPerFrame: 3,
            waveAmplitude: 10.0,  // Proper amplitude scaling for visualization

            // Cloth physics parameters
            clothGravity: 0.5,
            clothStiffness: 0.8,
            clothDamping: 0.95,
            motionSensitivity: 2.0,

            // Boundary inversion
            invertBoundaries: false,

            // Fluid simulation parameters (optimized for turbulence)
            fluidViscosity: 0.2,     // Higher base (shader scales down for turbulence)
            flowVelocity: 0.05,
            motionForce: 5.0,
            densityDisplay: 2.0,     // Increased for better visibility

            // Flow direction and strength
            flowDirection: 0.0,
            flowStrength: 0.0,

            // Schrödinger visualization parameters
            schrodingerDisplayMode: 0.0,  // 0=phase colors, 1=probability, 2=phase wheel
            schrodingerProbScale: 500.0,  // Scale factor for probability density (higher = brighter)
            schrodingerSigma: 30.0,       // Initial Gaussian width (localization)
            schrodingerKx: 0.8,           // Initial x-momentum
            schrodingerKy: 0.0,           // Initial y-momentum
            schrodingerVOffset: 0.0       // Potential offset
        };

        this.currentEquationType = 'wave';

        // Debug mode
        this.debugMode = false;
        this.frameCount = 0;

        // Motion detection state
        this.webcamFrameIndex = 0; // 0 or 1 for double buffering webcam frames

        // Fluid simulation state
        this.fluidBufferIndex = 0; // 0 or 1 for double buffering fluid fields
        this.velocityAdvectionPipeline = null;
        this.densityAdvectionPipeline = null;
        this.fluidBindGroupLayouts = {
            velocity: null,
            density: null
        };

        // Schrödinger simulation state
        this.schrodingerComputePipeline = null;
        this.schrodingerVisualizationPipeline = null;
    }

    async initialize() {
        await this.createBuffers();
        await this.createTextures();
        await this.createComputePipeline();
        this.initializeWaveField();

        console.log('Wave simulation initialized');
    }

    async createBuffers() {
        const bufferSize = this.width * this.height * 4; // 4 bytes per f32

        // Create triple wave field buffers for proper time stepping
        for (let i = 0; i < 3; i++) {
            this.buffers.waveBuffers[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Wave buffer ${i}`
            });
        }

        // Boundary buffer
        this.buffers.boundary = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // Webcam motion detection buffers (for motion-based sources)
        this.buffers.webcamPrev = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            label: 'Previous webcam frame'
        });

        this.buffers.webcamCurr = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            label: 'Current webcam frame'
        });

        // Fluid simulation buffers
        for (let i = 0; i < 2; i++) {
            this.buffers.velocityX[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Velocity X buffer ${i}`
            });

            this.buffers.velocityY[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Velocity Y buffer ${i}`
            });

            this.buffers.density[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Density buffer ${i}`
            });
        }

        this.buffers.pressure = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Pressure field'
        });

        this.buffers.divergence = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Velocity divergence'
        });

        // Schrödinger equation buffers
        for (let i = 0; i < 2; i++) {
            this.buffers.psiR[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Psi Real buffer ${i}`
            });

            this.buffers.psiI[i] = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: `Psi Imaginary buffer ${i}`
            });
        }

        this.buffers.potential = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Potential V(x,y)'
        });

        // Parameters uniform buffer
        // Layout: width(u32), height(u32), dt(f32), dx(f32), waveSpeed(f32), damping(f32), sourceFreq(f32), sourceEnabled(f32), time(f32), boundaryThreshold(f32), sourceStrength(f32), sourceSize(f32), normalizeWaves(f32), waveRange(f32), clothGravity(f32), clothStiffness(f32), clothDamping(f32), motionSensitivity(f32), invertBoundaries(f32), fluidViscosity(f32), flowVelocity(f32), motionForce(f32), densityDisplay(f32), flowDirection(f32), flowStrength(f32)
        const paramsSizeBytes = 108; // Size for all parameters including cloth, invert boundaries, LBM, and flow direction
        console.log(`Parameter buffer size: ${paramsSizeBytes} bytes for equation type: ${this.currentEquationType}`);
        this.buffers.params = this.device.createBuffer({
            size: paramsSizeBytes,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async createTextures() {
        // Output texture for rendering - using rgba16float to support full signed range
        this.textures.output = this.device.createTexture({
            size: [this.width, this.height, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST
        });

        this.textures.outputView = this.textures.output.createView();

        // Create buffer-to-texture conversion pipeline
        await this.createVisualizationPipeline();
    }

    async createVisualizationPipeline() {
        // Create a compute shader to convert buffer data to texture for visualization
        const visualizationShaderCode = `
@group(0) @binding(0) var<storage, read> wave_data: array<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: VisualizationParams;

struct VisualizationParams {
    width: u32,
    height: u32,
    time: f32,
    amplitude_scale: f32,
    value_offset: f32,
    _padding: vec3<f32>,  // Padding for alignment
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    let wave_value = wave_data[idx];
    
    // Apply offset first (to center Navier-Stokes density around 0), then scale
    // For wave equations: offset = 0.0 (no change)
    // For Navier-Stokes: offset = -6.0 to map [0,12] → [-6,6]
    let centered_value = wave_value + params.value_offset;
    let raw_amplitude = centered_value * params.amplitude_scale;
    
    // Store the raw amplitude in the red channel for the render pipeline
    // The render pipeline will apply the selected color scheme with proper scaling
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(raw_amplitude, 0.0, 0.0, 1.0));
}`;

        this.visualizationShader = this.device.createShaderModule({
            code: visualizationShaderCode
        });

        // Create bind group layout for visualization
        this.visualizationBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: 'write-only',
                        format: 'rgba16float'
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Create visualization pipeline
        this.visualizationPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.visualizationBindGroupLayout]
            }),
            compute: {
                module: this.visualizationShader,
                entryPoint: 'main'
            }
        });

        // Create visualization parameters buffer
        // Size must account for WebGPU uniform buffer alignment (16-byte boundaries)
        this.visualizationParamsBuffer = this.device.createBuffer({
            size: 48, // Uniform buffer with vec3 padding requires 48 bytes due to alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async createComputePipeline() {
        // Create bind group layout (standard for most shaders)
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Create motion bind group layout (for motion-based shader)
        this.motionBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Create identical layout for LBM (to avoid any subtle differences)
        this.lbmBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Fluid simulation bind group layouts
        this.fluidBindGroupLayouts.density = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velocity_x
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velocity_y
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // density
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // webcam_data
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }            // params
            ]
        });

        this.fluidBindGroupLayouts.velocity = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velocity_x_in
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // velocity_y_in
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // velocity_x_out
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // velocity_y_out
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // webcam_prev
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // webcam_curr
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }            // params
            ]
        });

        // Schrödinger equation bind group layout (7 bindings)
        this.schrodingerBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // psi_R (read)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // psi_I (read)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output (visualization)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // psi_R_next (write)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // psi_I_next (write)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // potential V(x,y)
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }            // params
            ]
        });

        // Schrödinger visualization bind group layout
        this.schrodingerVisualizationBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // psi_R
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // psi_I
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // output texture
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }            // params
            ]
        });

        await this.recompileShaders();
    }

    async recompileShaders() {
        try {
            const shaderCode = this.shaderManager.getShaderForEquationType(this.currentEquationType);

            const shaderModule = this.device.createShaderModule({
                code: shaderCode,
                label: `Wave shader - ${this.currentEquationType}`
            });

            // Wait for shader compilation to complete
            await shaderModule.getCompilationInfo().then(info => {
                if (info.messages.length > 0) {
                    console.warn('Shader compilation messages:', info.messages);
                    for (const message of info.messages) {
                        if (message.type === 'error') {
                            throw new Error(`Shader compilation error: ${message.message}`);
                        }
                    }
                }
            });

            // Choose the correct bind group layout based on equation type
            let layout;
            if (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') {
                layout = this.motionBindGroupLayout;
                console.log(`Using motion bind group layout for ${this.currentEquationType}`);
            } else if (this.currentEquationType === 'schrodinger') {
                layout = this.schrodingerBindGroupLayout;
                console.log(`Using Schrödinger bind group layout for ${this.currentEquationType}`);
            } else if (this.currentEquationType === 'navier-stokes-fluid') {
                layout = this.bindGroupLayout;
                console.log(`Using standard bind group layout for ${this.currentEquationType}`);
            } else {
                // Use standard layout for LBM and other equation types
                layout = this.bindGroupLayout;
                console.log(`Using standard bind group layout for ${this.currentEquationType}`);
            }

            // Create compute pipeline
            const layoutName = this.currentEquationType === 'lattice-boltzmann' ? 'LBM' :
                (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') ? 'motion' : 'standard';
            console.log(`Creating pipeline with ${layoutName} layout for ${this.currentEquationType}`);

            this.computePipeline = this.device.createComputePipeline({
                label: `Wave compute pipeline - ${this.currentEquationType}`,
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [layout]
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: 'main'
                }
            });

            console.log(`Pipeline created successfully for ${this.currentEquationType}`);

            // Simplified fluid simulation uses standard pipeline - no additional pipelines needed

            // Create bind groups
            this.createBindGroups();

        } catch (error) {
            console.error('Shader compilation error:', error);
            throw new Error(`Failed to compile shader: ${error.message}`);
        }
    }

    async createFluidPipelines() {
        try {
            console.log('Creating fluid pipelines...');

            // Create velocity advection pipeline
            const velocityShaderCode = this.shaderManager.getShaderForEquationType('velocity-advection');
            console.log('Got velocity shader code, length:', velocityShaderCode.length);

            const velocityShaderModule = this.device.createShaderModule({
                code: velocityShaderCode,
                label: 'Velocity advection shader'
            });

            // Wait for shader compilation
            await velocityShaderModule.getCompilationInfo().then(info => {
                if (info.messages.length > 0) {
                    console.warn('Velocity shader compilation messages:', info.messages);
                    for (const message of info.messages) {
                        if (message.type === 'error') {
                            throw new Error(`Velocity shader compilation error: ${message.message}`);
                        }
                    }
                }
            });

            console.log('Velocity shader compiled successfully');
            console.log('Fluid bind group layout for velocity:', this.fluidBindGroupLayouts.velocity);

            this.velocityAdvectionPipeline = this.device.createComputePipeline({
                label: 'Velocity advection pipeline',
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.fluidBindGroupLayouts.velocity]
                }),
                compute: {
                    module: velocityShaderModule,
                    entryPoint: 'main'
                }
            });

            console.log('Velocity advection pipeline created:', this.velocityAdvectionPipeline);
            console.log('Fluid pipelines created successfully');
        } catch (error) {
            console.error('Error creating fluid pipelines:', error);
            throw error;
        }
    }

    createBindGroups() {
        // We'll create the bind group dynamically each frame with correct buffer indices
        // This ensures proper buffer cycling for the wave equation
    }

    initializeWaveField() {
        const size = this.width * this.height;

        // Initialize all equation types with zeros first
        const zeros = new Float32Array(size);
        for (let i = 0; i < 3; i++) {
            this.device.queue.writeBuffer(this.buffers.waveBuffers[i], 0, zeros);
        }

        if (this.currentEquationType === 'lattice-boltzmann') {
            // LBM needs special initialization
            console.log('Initializing LBM with proper initial conditions');

            // u_prev: density field (start with uniform density ~1.0)
            const densityData = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                densityData[i] = 1.0; // Initial fluid density
            }
            this.device.queue.writeBuffer(this.buffers.waveBuffers[0], 0, densityData);

            // u_curr: velocity_x field (start with very small base flow)
            const velocityXData = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                velocityXData[i] = 0.001; // Very small initial flow so motion dominates
            }
            this.device.queue.writeBuffer(this.buffers.waveBuffers[1], 0, velocityXData);
        } else if (this.currentEquationType === 'navier-stokes-fluid') {
            // Initialize fluid simulation fields
            console.log('Initializing Navier-Stokes fluid simulation');

            // Initialize velocity fields with zero initial velocity
            const velocityXData = new Float32Array(size);
            const velocityYData = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                velocityXData[i] = 0.0; // No initial flow - let user control it
                velocityYData[i] = 0.0;
            }

            // Initialize density/dye field
            const densityData = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                densityData[i] = 0.0; // Start with no dye
            }

            // Write to both buffers
            for (let i = 0; i < 2; i++) {
                this.device.queue.writeBuffer(this.buffers.velocityX[i], 0, velocityXData);
                this.device.queue.writeBuffer(this.buffers.velocityY[i], 0, velocityYData);
                this.device.queue.writeBuffer(this.buffers.density[i], 0, densityData);
            }

            // Initialize pressure and divergence fields
            this.device.queue.writeBuffer(this.buffers.pressure, 0, zeros);
            this.device.queue.writeBuffer(this.buffers.divergence, 0, zeros);
        }

        // Schrödinger equation initialization
        if (this.currentEquationType === 'schrodinger') {
            console.log('Initializing Schrödinger equation with Gaussian wavepacket');

            // Initialize with a Gaussian wavepacket at center
            const psiRData = new Float32Array(size);
            const psiIData = new Float32Array(size);

            const centerX = this.width / 2;
            const centerY = this.height / 2;

            // Use UI parameters for wavepacket properties
            const sigma = this.params.schrodingerSigma || 30.0;  // Width of initial Gaussian (from UI)
            const k0x = (this.params.schrodingerKx || 0.8) * 0.3;  // Initial momentum in x (from UI)
            const k0y = (this.params.schrodingerKy || 0.0) * 0.3;  // Initial momentum in y (from UI)
            const amplitude = 1.0;

            console.log(`Schrödinger init: sigma=${sigma}, k0x=${k0x}, k0y=${k0y}`);
            console.log(`Raw params: schrodingerKx=${this.params.schrodingerKx}, schrodingerKy=${this.params.schrodingerKy}`);

            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    const idx = y * this.width + x;
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const r2 = dx * dx + dy * dy;

                    // Gaussian envelope
                    const envelope = amplitude * Math.exp(-r2 / (2 * sigma * sigma));

                    // Phase from initial momentum
                    const phase = k0x * dx + k0y * dy;

                    // Complex wavefunction: ψ = envelope * exp(i*phase)
                    psiRData[idx] = envelope * Math.cos(phase);
                    psiIData[idx] = envelope * Math.sin(phase);
                }
            }

            // Write to both buffer sets
            for (let i = 0; i < 2; i++) {
                this.device.queue.writeBuffer(this.buffers.psiR[i], 0, psiRData);
                this.device.queue.writeBuffer(this.buffers.psiI[i], 0, psiIData);
            }

            // Initialize potential buffer with zeros (will be filled from webcam)
            this.device.queue.writeBuffer(this.buffers.potential, 0, zeros);

            this.schrodingerBufferIndex = 0;
        }

        // Initialize boundary and webcam buffers with zeros for all equation types
        this.device.queue.writeBuffer(this.buffers.boundary, 0, zeros);
        this.device.queue.writeBuffer(this.buffers.webcamPrev, 0, zeros);
        this.device.queue.writeBuffer(this.buffers.webcamCurr, 0, zeros);

        this.updateParametersBuffer();
    }

    updateParameters(newParams) {
        let updated = false;

        for (const [key, value] of Object.entries(newParams)) {
            if (key in this.params && this.params[key] !== value) {
                // Debug logging for flow parameters
                if (key === 'flowDirection' || key === 'flowStrength' || key === 'motionForce') {
                    console.log(`WaveSimulation: ${key} updated from ${this.params[key]} to ${value}`);
                }
                this.params[key] = value;
                updated = true;
            }
        }

        if (updated) {
            this.updateParametersBuffer();
        }
    }

    updateParametersBuffer() {
        // Pack parameters into a buffer with correct types
        const buffer = new ArrayBuffer(108); // Increased for cloth, invert boundaries, LBM, and flow direction parameters
        const view = new DataView(buffer);
        let offset = 0;
        view.setUint32(offset, this.params.width, true); offset += 4;
        view.setUint32(offset, this.params.height, true); offset += 4;
        view.setFloat32(offset, this.params.dt, true); offset += 4;
        view.setFloat32(offset, this.params.dx, true); offset += 4;
        view.setFloat32(offset, this.params.waveSpeed, true); offset += 4;
        view.setFloat32(offset, this.params.damping, true); offset += 4;
        view.setFloat32(offset, this.params.sourceFrequency, true); offset += 4;
        view.setFloat32(offset, this.params.sourceEnabled ? 1.0 : 0.0, true); offset += 4;
        view.setFloat32(offset, this.params.time, true); offset += 4;
        view.setFloat32(offset, this.params.boundaryThreshold, true); offset += 4;
        view.setFloat32(offset, this.params.sourceStrength || 0.5, true); offset += 4;
        view.setFloat32(offset, this.params.sourceSize || 20.0, true); offset += 4;
        view.setFloat32(offset, this.params.normalizeWaves ? 1.0 : 0.0, true); offset += 4;
        view.setFloat32(offset, this.params.waveRange || 1.0, true); offset += 4;

        // Cloth physics parameters
        view.setFloat32(offset, this.params.clothGravity || 0.5, true); offset += 4;
        view.setFloat32(offset, this.params.clothStiffness || 0.8, true); offset += 4;
        view.setFloat32(offset, this.params.clothDamping || 0.95, true); offset += 4;
        view.setFloat32(offset, this.params.motionSensitivity || 2.0, true); offset += 4;

        // Boundary inversion
        view.setFloat32(offset, this.params.invertBoundaries ? 1.0 : 0.0, true); offset += 4;

        // Lattice Boltzmann parameters
        view.setFloat32(offset, this.params.fluidViscosity, true); offset += 4;
        view.setFloat32(offset, this.params.flowVelocity, true); offset += 4;
        view.setFloat32(offset, this.params.motionForce, true); offset += 4;
        view.setFloat32(offset, this.params.densityDisplay, true); offset += 4;

        // Flow direction and strength parameters
        const flowDir = this.params.flowDirection;
        const flowStr = this.params.flowStrength;
        view.setFloat32(offset, flowDir, true); offset += 4;
        view.setFloat32(offset, flowStr, true); offset += 4;

        // Debug logging for parameter buffer writing
        if (this.currentEquationType === 'navier-stokes-fluid' && (flowStr > 0.01 || this.frameCount % 120 === 0)) {
            console.log(`Buffer: flowDirection=${flowDir}, flowStrength=${flowStr}, motionForce=${this.params.motionForce}`);
        }

        this.device.queue.writeBuffer(this.buffers.params, 0, buffer);
    }

    updateBoundaryFromWebcam(webcamTexture) {
        if (!webcamTexture) {
            // Create boundary at edges only if no webcam
            const boundaryData = new Float32Array(this.width * this.height);
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    const idx = y * this.width + x;
                    if (x === 0 || x === this.width - 1 || y === 0 || y === this.height - 1) {
                        boundaryData[idx] = this.currentEquationType === 'wave-bathymetry' ? 0.0 : 1.0;
                    } else {
                        boundaryData[idx] = this.currentEquationType === 'wave-bathymetry' ? 0.5 : 0.0;
                    }
                }
            }
            this.device.queue.writeBuffer(this.buffers.boundary, 0, boundaryData);
            return;
        }

        // Use different extraction modes based on equation type
        if (this.currentEquationType === 'wave-bathymetry') {
            this.extractDepthFromWebcam(webcamTexture);
        } else if (this.currentEquationType === 'schrodinger') {
            // For Schrödinger, extract webcam intensity as potential V(x,y)
            this.extractPotentialFromWebcam(webcamTexture);
        } else if (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') {
            this.updateWebcamMotionBuffers(webcamTexture);
        } else if (this.currentEquationType === 'navier-stokes-fluid') {
            // Navier-Stokes supports both motion detection and simple intensity
            // Use motion_sensitivity parameter to control mode:
            // - High values (> 1.5): motion detection mode
            // - Low values (<= 1.5): simple intensity mode
            if (this.params.motionSensitivity > 1.5) {
                // Motion detection mode (differential frames)
                this.updateWebcamMotionBuffers(webcamTexture);
                this.extractMotionToBoundary();
            } else {
                // Simple intensity mode (like wave-bathymetry but for fluid)
                this.extractIntensityFromWebcam(webcamTexture);
            }
        } else {
            // LBM and other equation types use standard boundary extraction
            this.extractBoundariesFromWebcam(webcamTexture);
        }
    }

    extractBoundariesFromWebcam(webcamTexture) {
        // Create boundary extraction compute shader if it doesn't exist
        if (!this.boundaryExtractionPipeline) {
            this.createBoundaryExtractionPipeline();
        }

        // Create bind group for boundary extraction
        const boundaryBindGroup = this.device.createBindGroup({
            layout: this.boundaryExtractionPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.boundary }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.params }
                }
            ]
        });

        // Create command encoder for boundary extraction
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Boundary extraction command encoder'
        });

        // Run boundary extraction compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Boundary extraction compute pass'
        });

        computePass.setPipeline(this.boundaryExtractionPipeline);
        computePass.setBindGroup(0, boundaryBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit boundary extraction commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    extractDepthFromWebcam(webcamTexture) {
        // Create depth extraction compute shader if it doesn't exist
        if (!this.depthExtractionPipeline) {
            this.createDepthExtractionPipeline();
        }

        // Create bind group for depth extraction
        const depthBindGroup = this.device.createBindGroup({
            layout: this.depthExtractionPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.boundary }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.params }
                }
            ]
        });

        // Create command encoder for depth extraction
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Depth extraction command encoder'
        });

        // Run depth extraction compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Depth extraction compute pass'
        });

        computePass.setPipeline(this.depthExtractionPipeline);
        computePass.setBindGroup(0, depthBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit depth extraction commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    extractIntensityFromWebcam(webcamTexture) {
        // Create intensity extraction compute shader if it doesn't exist
        if (!this.intensityExtractionPipeline) {
            this.createIntensityExtractionPipeline();
        }

        // Create bind group for intensity extraction
        const intensityBindGroup = this.device.createBindGroup({
            layout: this.intensityExtractionPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.boundary }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.params }
                }
            ]
        });

        // Create command encoder for intensity extraction
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Intensity extraction command encoder'
        });

        // Run intensity extraction compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Intensity extraction compute pass'
        });

        computePass.setPipeline(this.intensityExtractionPipeline);
        computePass.setBindGroup(0, intensityBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit intensity extraction commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    createIntensityExtractionPipeline() {
        const intensityExtractionShaderCode = `
@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> intensity_data: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,
    damping: f32,
    source_freq: f32,
    source_enabled: f32,
    time: f32,
    boundary_threshold: f32,
    source_strength: f32,
    source_size: f32,
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,
    cloth_stiffness: f32,
    cloth_damping: f32,
    motion_sensitivity: f32,
    invert_boundaries: f32,
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32,
    flow_direction: f32,
    flow_strength: f32
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Sample webcam texture
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    
    // Convert to grayscale using luminance formula
    let intensity = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    // Store intensity directly (0.0 to 1.0 range)
    // This will be used as dye source in the fluid shader
    intensity_data[idx] = intensity;
}`;

        const shaderModule = this.device.createShaderModule({
            code: intensityExtractionShaderCode,
            label: 'Intensity extraction shader'
        });

        this.intensityExtractionPipeline = this.device.createComputePipeline({
            label: 'Intensity extraction pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    extractPotentialFromWebcam(webcamTexture) {
        // Create potential extraction pipeline if it doesn't exist
        if (!this.potentialExtractionPipeline) {
            this.createPotentialExtractionPipeline();
        }

        // Create bind group for potential extraction
        const potentialBindGroup = this.device.createBindGroup({
            layout: this.potentialExtractionPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.potential }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.params }
                }
            ]
        });

        // Create command encoder for potential extraction
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Potential extraction command encoder'
        });

        // Run potential extraction compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Potential extraction compute pass'
        });

        computePass.setPipeline(this.potentialExtractionPipeline);
        computePass.setBindGroup(0, potentialBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit potential extraction commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    createPotentialExtractionPipeline() {
        const potentialExtractionShaderCode = `
@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> potential_data: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,
    damping: f32,
    source_freq: f32,
    source_enabled: f32,
    time: f32,
    boundary_threshold: f32,
    source_strength: f32,
    source_size: f32,
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,
    cloth_stiffness: f32,
    cloth_damping: f32,
    motion_sensitivity: f32,
    invert_boundaries: f32,
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32,
    flow_direction: f32,
    flow_strength: f32
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Sample webcam texture
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    
    // Convert to grayscale using luminance formula
    let intensity = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    // Apply threshold to create potential barriers
    // Dark areas (below threshold) create high potential barriers
    // Bright areas have low/zero potential
    var potential: f32;
    if (params.invert_boundaries > 0.5) {
        // Bright areas become barriers
        potential = intensity;
    } else {
        // Dark areas become barriers (default)
        potential = 1.0 - intensity;
    }
    
    // Apply non-linear scaling for sharper potential edges
    potential = pow(potential, 2.0);
    
    // Scale by boundary threshold (basic normalization)
    // Further scaling is done in the Schrödinger shader via amplitude parameter
    potential = potential * params.boundary_threshold;
    
    potential_data[idx] = potential;
}`;

        const shaderModule = this.device.createShaderModule({
            code: potentialExtractionShaderCode,
            label: 'Potential extraction shader'
        });

        this.potentialExtractionPipeline = this.device.createComputePipeline({
            label: 'Potential extraction pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    createDepthExtractionPipeline() {
        const depthExtractionShaderCode = `
@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> depth_data: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,
    damping: f32,
    source_freq: f32,
    source_enabled: f32,
    time: f32,
    boundary_threshold: f32,
    source_strength: f32,
    source_size: f32,
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,
    cloth_stiffness: f32,
    cloth_damping: f32,
    motion_sensitivity: f32,
    invert_boundaries: f32,
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Sample webcam texture
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    
    // Convert to grayscale using luminance formula
    let grayscale = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    // Convert grayscale to depth: dark = shallow (0.0), bright = deep (1.0)
    // Apply some smoothing and contrast adjustment
    let depth = pow(grayscale, 0.8); // Slight gamma correction for better contrast
    
    depth_data[idx] = depth;
    
    // Keep screen edges as shallow water for stability
    if (x == 0 || x == params.width - 1 || y == 0 || y == params.height - 1) {
        depth_data[idx] = 0.1; // Shallow edges
    }
}`;

        const shaderModule = this.device.createShaderModule({
            code: depthExtractionShaderCode,
            label: 'Depth extraction shader'
        });

        this.depthExtractionPipeline = this.device.createComputePipeline({
            label: 'Depth extraction pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    createBoundaryExtractionPipeline() {
        const boundaryExtractionShaderCode = `
@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> boundary_data: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,
    damping: f32,
    source_freq: f32,
    source_enabled: f32,
    time: f32,
    boundary_threshold: f32,
    source_strength: f32,
    source_size: f32,
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,
    cloth_stiffness: f32,
    cloth_damping: f32,
    motion_sensitivity: f32,
    invert_boundaries: f32,
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Sample webcam texture
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    
    // Convert to grayscale using luminance formula
    let grayscale = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    // Create boundary based on threshold
    // Dark areas (below threshold) become boundaries
    if (grayscale < params.boundary_threshold) {
        boundary_data[idx] = 1.0; // Boundary
    } else {
        boundary_data[idx] = 0.0; // Free space
    }
    
    // Always keep screen edges as boundaries
    if (x == 0 || x == params.width - 1 || y == 0 || y == params.height - 1) {
        boundary_data[idx] = 1.0;
    }
}`;

        const shaderModule = this.device.createShaderModule({
            code: boundaryExtractionShaderCode,
            label: 'Boundary extraction shader'
        });

        this.boundaryExtractionPipeline = this.device.createComputePipeline({
            label: 'Boundary extraction pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    extractMotionToBoundary() {
        // Create motion detection pipeline if it doesn't exist
        if (!this.motionDetectionPipeline) {
            this.createMotionDetectionPipeline();
        }

        // Create bind group for motion detection
        const motionBindGroup = this.device.createBindGroup({
            layout: this.motionDetectionPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.buffers.webcamPrev }
                },
                {
                    binding: 1,
                    resource: { buffer: this.buffers.webcamCurr }
                },
                {
                    binding: 2,
                    resource: { buffer: this.buffers.boundary }
                },
                {
                    binding: 3,
                    resource: { buffer: this.buffers.params }
                }
            ]
        });

        // Create command encoder for motion detection
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Motion detection command encoder'
        });

        // Run motion detection compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Motion detection compute pass'
        });

        computePass.setPipeline(this.motionDetectionPipeline);
        computePass.setBindGroup(0, motionBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit motion detection commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    createMotionDetectionPipeline() {
        const motionDetectionShaderCode = `
@group(0) @binding(0) var<storage, read> webcam_prev: array<f32>;
@group(0) @binding(1) var<storage, read> webcam_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> motion_data: array<f32>;
@group(0) @binding(3) var<uniform> params: SimParams;

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,
    damping: f32,
    source_freq: f32,
    source_enabled: f32,
    time: f32,
    boundary_threshold: f32,
    source_strength: f32,
    source_size: f32,
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,
    cloth_stiffness: f32,
    cloth_damping: f32,
    motion_sensitivity: f32,
    invert_boundaries: f32,
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32,
    flow_direction: f32,
    flow_strength: f32
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Calculate temporal difference (motion) - square of the difference as requested
    let prev_intensity = webcam_prev[idx];
    let curr_intensity = webcam_curr[idx];
    let motion_diff = curr_intensity - prev_intensity;
    let motion_squared = motion_diff * motion_diff; // Square of the difference per pixel
    
    // Apply motion sensitivity parameter with enhanced scaling
    var enhanced_motion = motion_squared * params.motion_force * 25.0;
    
    // Add some base sensitivity for very small motions
    if (motion_squared > 0.0001) {
        enhanced_motion += 0.1; // Base motion boost
    }
    
    // Clamp to reasonable range for visibility
    enhanced_motion = clamp(enhanced_motion, 0.0, 10.0);
    
    motion_data[idx] = enhanced_motion;
}`;

        const shaderModule = this.device.createShaderModule({
            code: motionDetectionShaderCode,
            label: 'Motion detection shader'
        });

        this.motionDetectionPipeline = this.device.createComputePipeline({
            label: 'Motion detection pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    updateWebcamMotionBuffers(webcamTexture) {
        // Double buffer the webcam frames for motion detection
        if (this.webcamFrameIndex === 0) {
            // Extract to current buffer
            this.extractWebcamToBuffer(webcamTexture, this.buffers.webcamCurr);
        } else {
            // Extract to previous buffer (which becomes the new current)
            this.extractWebcamToBuffer(webcamTexture, this.buffers.webcamPrev);
        }

        // Swap frame index for next update
        this.webcamFrameIndex = 1 - this.webcamFrameIndex;
    }

    extractWebcamToBuffer(webcamTexture, targetBuffer) {
        // Create webcam-to-buffer extraction shader if it doesn't exist
        if (!this.webcamToBufferPipeline) {
            this.createWebcamToBufferPipeline();
        }

        // Create bind group for webcam extraction
        const webcamBindGroup = this.device.createBindGroup({
            layout: this.webcamToBufferPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: webcamTexture.createView()
                },
                {
                    binding: 1,
                    resource: { buffer: targetBuffer }
                }
            ]
        });

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Webcam to buffer extraction'
        });

        // Run extraction compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Webcam extraction compute pass'
        });

        computePass.setPipeline(this.webcamToBufferPipeline);
        computePass.setBindGroup(0, webcamBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    createWebcamToBufferPipeline() {
        const webcamToBufferShaderCode = `
@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> webcam_buffer: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Get texture dimensions
    let dims = textureDimensions(webcam_texture);
    
    if (x >= dims.x || y >= dims.y) {
        return;
    }
    
    let idx = y * dims.x + x;
    
    // Sample webcam texture and convert to grayscale
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let grayscale = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    webcam_buffer[idx] = grayscale;
}`;

        const shaderModule = this.device.createShaderModule({
            code: webcamToBufferShaderCode,
            label: 'Webcam to buffer shader'
        });

        this.webcamToBufferPipeline = this.device.createComputePipeline({
            label: 'Webcam to buffer pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    step(webcamTexture) {
        try {
            // Update boundary conditions from webcam (only if webcam texture is provided)
            if (webcamTexture) {
                this.updateBoundaryFromWebcam(webcamTexture);
            }

            // Update time
            this.time += this.dt;
            this.params.time = this.time;
            this.frameCount++;
            this.updateParametersBuffer();

            // Validate compute pipeline exists
            if (!this.computePipeline) {
                console.error('Compute pipeline is null, skipping frame');
                return;
            }

            // Navier-Stokes now uses standard pipeline approach (no special handling needed)

            // Create bind group with correct buffer cycling
            let bindGroup;

            // Special handling for Schrödinger equation
            if (this.currentEquationType === 'schrodinger') {
                this.runSchrodingerStep();
                return; // Schrödinger has its own step logic
            }

            if (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') {
                // Motion-based shader needs webcam buffers
                const webcamPrevBuffer = this.webcamFrameIndex === 0 ? this.buffers.webcamPrev : this.buffers.webcamCurr;
                const webcamCurrBuffer = this.webcamFrameIndex === 0 ? this.buffers.webcamCurr : this.buffers.webcamPrev;

                /*console.log(`🌊 MOTION SHADER ACTIVE for ${this.currentEquationType} with buffers:`, {
                    prev: this.bufferIndices.prev,
                    curr: this.bufferIndices.curr,
                    next: this.bufferIndices.next,
                    webcamIndex: this.webcamFrameIndex,
                    motionForce: this.params?.motionForce || 'undefined',
                    sourceStrength: this.params?.sourceStrength || 'undefined'
                });*/

                try {
                    /*console.log('Creating motion bind group with resources:', {
                        waveBuffers: !!this.buffers.waveBuffers,
                        webcamPrevBuffer: !!webcamPrevBuffer,
                        webcamCurrBuffer: !!webcamCurrBuffer,
                        paramsBuffer: !!this.buffers.params,
                        motionBindGroupLayout: !!this.motionBindGroupLayout
                    });*/

                    bindGroup = this.device.createBindGroup({
                        layout: this.motionBindGroupLayout,
                        entries: [
                            { binding: 0, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.prev] } },
                            { binding: 1, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.curr] } },
                            { binding: 2, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.next] } },
                            { binding: 3, resource: { buffer: webcamPrevBuffer } },
                            { binding: 4, resource: { buffer: webcamCurrBuffer } },
                            { binding: 5, resource: { buffer: this.buffers.params } }
                        ]
                    });
                    //console.log('✅ Motion bind group created successfully');
                } catch (error) {
                    console.error('❌ Failed to create motion bind group:', error);
                    console.log('Buffer details:', {
                        prevBuffer: this.buffers.waveBuffers[this.bufferIndices.prev],
                        currBuffer: this.buffers.waveBuffers[this.bufferIndices.curr],
                        nextBuffer: this.buffers.waveBuffers[this.bufferIndices.next],
                        webcamPrev: webcamPrevBuffer,
                        webcamCurr: webcamCurrBuffer,
                        params: this.buffers.params
                    });
                    return;
                }
            } else {
                // Standard shader uses boundary buffer
                bindGroup = this.device.createBindGroup({
                    layout: this.bindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.prev] } },
                        { binding: 1, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.curr] } },
                        { binding: 2, resource: { buffer: this.buffers.waveBuffers[this.bufferIndices.next] } },
                        { binding: 3, resource: { buffer: this.buffers.boundary } },
                        { binding: 4, resource: { buffer: this.buffers.params } }
                    ]
                });
            }

            // Create command encoder
            const commandEncoder = this.device.createCommandEncoder({
                label: 'Wave simulation command encoder'
            });

            // Compute pass
            const computePass = commandEncoder.beginComputePass({
                label: 'Wave simulation compute pass'
            });

            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, bindGroup);

            // Dispatch compute shader
            const workgroupsX = Math.ceil(this.width / 8);
            const workgroupsY = Math.ceil(this.height / 8);
            computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

            computePass.end();

            // Copy result to output texture
            this.copyBufferToTexture(commandEncoder);

            // Submit commands
            const commandBuffer = commandEncoder.finish();
            this.device.queue.submit([commandBuffer]);

            // Properly cycle buffers: prev <- curr <- next
            const temp = this.bufferIndices.prev;
            this.bufferIndices.prev = this.bufferIndices.curr;
            this.bufferIndices.curr = this.bufferIndices.next;
            this.bufferIndices.next = temp;

        } catch (error) {
            console.error('Error in wave simulation step:', error);
            // Try to recover by reinitializing the pipeline
            this.recompileShaders().catch(e => {
                console.error('Failed to recover from simulation error:', e);
            });
        }
    }

    runFluidSimulationStep(webcamTexture, commandEncoder) {
        try {
            // Validate that fluid pipelines exist
            if (!this.velocityAdvectionPipeline) {
                console.error('Velocity advection pipeline is null, cannot run fluid simulation');
                console.log('Current equation type:', this.currentEquationType);
                console.log('All pipelines:', {
                    computePipeline: this.computePipeline,
                    velocityAdvectionPipeline: this.velocityAdvectionPipeline,
                    motionDetectionPipeline: this.motionDetectionPipeline
                });
                return;
            }

            // Webcam frames were already updated in updateBoundaryFromWebcam
            // Do not update them again here to preserve correct prev/curr pairing

            const workgroupsX = Math.ceil(this.width / 8);
            const workgroupsY = Math.ceil(this.height / 8);

            // Step 1: Velocity advection
            const velocityBindGroup = this.device.createBindGroup({
                layout: this.fluidBindGroupLayouts.velocity,
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.velocityX[this.fluidBufferIndex] } },
                    { binding: 1, resource: { buffer: this.buffers.velocityY[this.fluidBufferIndex] } },
                    { binding: 2, resource: { buffer: this.buffers.velocityX[1 - this.fluidBufferIndex] } },
                    { binding: 3, resource: { buffer: this.buffers.velocityY[1 - this.fluidBufferIndex] } },
                    // After updateWebcamMotionBuffers toggles the index:
                    // webcamFrameIndex=0 means latest frame is in webcamPrev, previous is in webcamCurr
                    // webcamFrameIndex=1 means latest frame is in webcamCurr, previous is in webcamPrev
                    { binding: 4, resource: { buffer: this.webcamFrameIndex === 0 ? this.buffers.webcamCurr : this.buffers.webcamPrev } }, // prev
                    { binding: 5, resource: { buffer: this.webcamFrameIndex === 0 ? this.buffers.webcamPrev : this.buffers.webcamCurr } }, // curr
                    { binding: 6, resource: { buffer: this.buffers.params } }
                ]
            });

            const velocityPass = commandEncoder.beginComputePass({ label: 'Velocity advection pass' });
            velocityPass.setPipeline(this.velocityAdvectionPipeline);
            velocityPass.setBindGroup(0, velocityBindGroup);
            velocityPass.dispatchWorkgroups(workgroupsX, workgroupsY);
            velocityPass.end();

            // Step 2: Density advection
            const densityBindGroup = this.device.createBindGroup({
                layout: this.fluidBindGroupLayouts.density,
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.velocityX[1 - this.fluidBufferIndex] } },
                    { binding: 1, resource: { buffer: this.buffers.velocityY[1 - this.fluidBufferIndex] } },
                    { binding: 2, resource: { buffer: this.buffers.density[1 - this.fluidBufferIndex] } },
                    { binding: 3, resource: { buffer: this.buffers.boundary } },
                    { binding: 4, resource: { buffer: this.buffers.params } }
                ]
            });

            const densityPass = commandEncoder.beginComputePass({ label: 'Density advection pass' });
            densityPass.setPipeline(this.computePipeline);
            densityPass.setBindGroup(0, densityBindGroup);
            densityPass.dispatchWorkgroups(workgroupsX, workgroupsY);
            densityPass.end();

            // Copy density result to output texture
            this.copyFluidToTexture(commandEncoder);

            // Submit commands
            const commandBuffer = commandEncoder.finish();
            this.device.queue.submit([commandBuffer]);

            // Swap fluid buffers only (webcam buffers are managed in updateBoundaryFromWebcam)
            this.fluidBufferIndex = 1 - this.fluidBufferIndex;

        } catch (error) {
            console.error('Error in fluid simulation step:', error);
        }
    }

    copyFluidToTexture(commandEncoder) {
        // Update visualization parameters
        const viz = new ArrayBuffer(48); // Must match buffer size (48 bytes)
        const vizView = new DataView(viz);
        vizView.setUint32(0, this.width, true);
        vizView.setUint32(4, this.height, true);
        vizView.setFloat32(8, this.time, true);

        // Adaptive amplitude scaling and offset based on equation type
        let amplitudeScale = this.params.waveAmplitude || 5.0;
        let valueOffset = 0.0; // Default: no offset for wave equations

        if (this.currentEquationType === 'lattice-boltzmann') {
            amplitudeScale = 2.0; // LBM produces larger values
            valueOffset = -5.0; // Center LBM density around 0
        } else if (this.currentEquationType === 'navier-stokes-fluid') {
            amplitudeScale = 1.5; // Adjusted for render pipeline color schemes
            valueOffset = -6.0; // Center NS density [0,12] → [-6,6]
        } else if (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') {
            amplitudeScale = 10.0; // Motion-based simulations produce smaller values
            valueOffset = 0.0; // No offset needed
        }

        vizView.setFloat32(12, amplitudeScale, true);
        vizView.setFloat32(16, valueOffset, true);
        // Remaining bytes 20-47 are left as zeros for padding
        this.device.queue.writeBuffer(this.visualizationParamsBuffer, 0, viz);

        // Use the current density buffer for visualization
        const currentDensityBuffer = this.buffers.density[1 - this.fluidBufferIndex];

        // Create bind group for visualization
        const visualizationBindGroup = this.device.createBindGroup({
            layout: this.visualizationBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: currentDensityBuffer } },
                { binding: 1, resource: this.textures.outputView },
                { binding: 2, resource: { buffer: this.visualizationParamsBuffer } }
            ]
        });

        // Run visualization compute pass
        const visualizationPass = commandEncoder.beginComputePass({ label: 'Fluid visualization pass' });
        visualizationPass.setPipeline(this.visualizationPipeline);
        visualizationPass.setBindGroup(0, visualizationBindGroup);
        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        visualizationPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        visualizationPass.end();
    }

    copyBufferToTexture(commandEncoder) {
        if (this.debugMode) {
            // Create a simple test pattern to verify rendering pipeline
            this.createDebugPattern(commandEncoder);
            return;
        }

        // Update visualization parameters with correct types
        const viz = new ArrayBuffer(48); // Must match buffer size (48 bytes)
        const vizView = new DataView(viz);
        vizView.setUint32(0, this.width, true);
        vizView.setUint32(4, this.height, true);
        vizView.setFloat32(8, this.time, true);

        // Adaptive amplitude scaling and offset based on equation type
        let amplitudeScale = this.params.waveAmplitude || 5.0;
        let valueOffset = 0.0; // Default: no offset for wave equations

        if (this.currentEquationType === 'lattice-boltzmann') {
            amplitudeScale = 2.0; // LBM produces larger values
            valueOffset = -5.0; // Center LBM density around 0
        } else if (this.currentEquationType === 'navier-stokes-fluid') {
            amplitudeScale = 1.5; // Adjusted for render pipeline color schemes
            valueOffset = -6.0; // Center NS density [0,12] → [-6,6]
        } else if (this.currentEquationType === 'wave-motion' || this.currentEquationType === 'cloth-drapery') {
            amplitudeScale = 10.0; // Motion-based simulations produce smaller values
            valueOffset = 0.0; // No offset needed
        }

        vizView.setFloat32(12, amplitudeScale, true);
        vizView.setFloat32(16, valueOffset, true);
        // Remaining bytes 20-47 are left as zeros for padding
        this.device.queue.writeBuffer(this.visualizationParamsBuffer, 0, viz);

        // Use the current buffer (most recent computed values)
        const latestBuffer = this.buffers.waveBuffers[this.bufferIndices.curr];

        // Create bind group for visualization
        const visualizationBindGroup = this.device.createBindGroup({
            layout: this.visualizationBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: latestBuffer } },
                { binding: 1, resource: this.textures.outputView },
                { binding: 2, resource: { buffer: this.visualizationParamsBuffer } }
            ]
        });

        // Run visualization compute pass
        const visualizationPass = commandEncoder.beginComputePass();
        visualizationPass.setPipeline(this.visualizationPipeline);
        visualizationPass.setBindGroup(0, visualizationBindGroup);
        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        visualizationPass.dispatchWorkgroups(workgroupsX, workgroupsY);
        visualizationPass.end();
    }

    createDebugPattern(commandEncoder) {
        // Create a simple animated debug pattern to verify the rendering pipeline
        const debugShaderCode = `
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: DebugParams;

struct DebugParams {
    width: u32,
    height: u32,
    time: f32,
    frame_count: f32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // Create animated test pattern
    let fx = f32(x) / f32(params.width);
    let fy = f32(y) / f32(params.height);
    
    // Moving sine waves
    let wave1 = sin(fx * 10.0 + params.time * 2.0) * 0.5 + 0.5;
    let wave2 = sin(fy * 8.0 + params.time * 1.5) * 0.5 + 0.5;
    let wave3 = sin((fx + fy) * 6.0 + params.time * 3.0) * 0.5 + 0.5;
    
    // Pulsing center
    let center_x = f32(params.width) * 0.5;
    let center_y = f32(params.height) * 0.5;
    let dist = distance(vec2<f32>(f32(x), f32(y)), vec2<f32>(center_x, center_y));
    let pulse = sin(params.time * 4.0 - dist * 0.1) * 0.3 + 0.7;
    
    let color = vec3<f32>(
        wave1 * pulse,
        wave2 * pulse, 
        wave3 * pulse
    );
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}`;

        // Create debug shader if it doesn't exist
        if (!this.debugShader) {
            this.debugShader = this.device.createShaderModule({
                code: debugShaderCode
            });

            // Create debug pipeline
            this.debugPipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: this.debugShader,
                    entryPoint: 'main'
                }
            });

            // Create debug params buffer
            this.debugParamsBuffer = this.device.createBuffer({
                size: 4 * 4, // 4 f32 values
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
        }

        // Update debug parameters
        const debugParams = new Float32Array([
            this.width,
            this.height,
            this.time,
            this.frameCount
        ]);

        this.device.queue.writeBuffer(this.debugParamsBuffer, 0, debugParams);

        // Create bind group for debug
        const debugBindGroup = this.device.createBindGroup({
            layout: this.debugPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.textures.outputView
                },
                {
                    binding: 1,
                    resource: { buffer: this.debugParamsBuffer }
                }
            ]
        });

        // Run debug compute pass
        const debugPass = commandEncoder.beginComputePass();
        debugPass.setPipeline(this.debugPipeline);
        debugPass.setBindGroup(0, debugBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        debugPass.dispatchWorkgroups(workgroupsX, workgroupsY);

        debugPass.end();

        console.log(`Debug frame ${this.frameCount}: time=${this.time.toFixed(2)}`);
    }

    runSchrodingerStep() {
        // Schrödinger equation simulation step
        // Uses dedicated buffers for complex wavefunction (psiR, psiI)

        // Ensure Schrödinger pipeline exists
        if (!this.schrodingerComputePipeline) {
            this.createSchrodingerComputePipeline();
        }

        const curr = this.schrodingerBufferIndex;
        const next = 1 - curr;

        // Create bind group for Schrödinger computation
        const bindGroup = this.device.createBindGroup({
            layout: this.schrodingerBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiR[curr] } },      // psi_R input
                { binding: 1, resource: { buffer: this.buffers.psiI[curr] } },      // psi_I input
                { binding: 2, resource: { buffer: this.buffers.waveBuffers[0] } },  // output for visualization
                { binding: 3, resource: { buffer: this.buffers.psiR[next] } },      // psi_R output
                { binding: 4, resource: { buffer: this.buffers.psiI[next] } },      // psi_I output
                { binding: 5, resource: { buffer: this.buffers.potential } },       // V(x,y) potential
                { binding: 6, resource: { buffer: this.buffers.params } }           // params
            ]
        });

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Schrödinger simulation command encoder'
        });

        // Run main Schrödinger compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Schrödinger compute pass'
        });

        computePass.setPipeline(this.schrodingerComputePipeline);
        computePass.setBindGroup(0, bindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);

        computePass.end();

        // Now run the visualization pass to create the colorful output
        this.runSchrodingerVisualization(commandEncoder, next);

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Swap buffers
        this.schrodingerBufferIndex = next;
    }

    runSchrodingerVisualization(commandEncoder, bufferIndex) {
        // Create visualization pipeline if it doesn't exist
        if (!this.schrodingerVisualizationPipeline) {
            this.createSchrodingerVisualizationPipeline();
        }

        // Update visualization parameters
        // Struct layout (all naturally aligned, 32 bytes total):
        // width: u32 at offset 0
        // height: u32 at offset 4
        // time: f32 at offset 8
        // amplitude_scale: f32 at offset 12
        // gamma: f32 at offset 16
        // display_mode: f32 at offset 20 (0=phase, 1=probability, 2=phase wheel)
        // prob_scale: f32 at offset 24
        // _padding: f32 at offset 28
        const vizParams = new ArrayBuffer(32);
        const vizView = new DataView(vizParams);
        vizView.setUint32(0, this.width, true);
        vizView.setUint32(4, this.height, true);
        vizView.setFloat32(8, this.time, true);
        vizView.setFloat32(12, this.params.waveAmplitude || 5.0, true);   // amplitude_scale
        vizView.setFloat32(16, this.params.motionSensitivity || 0.6, true);  // gamma for contrast
        vizView.setFloat32(20, this.params.schrodingerDisplayMode || 0.0, true);  // display_mode
        vizView.setFloat32(24, this.params.schrodingerProbScale || 100.0, true);  // prob_scale
        vizView.setFloat32(28, 0.0, true);  // padding

        this.device.queue.writeBuffer(this.schrodingerVisualizationParamsBuffer, 0, vizParams);

        // Debug: Log visualization params occasionally
        //if (this.frameCount % 60 === 0) {
        //    console.log(`Schrodinger viz: mode=${this.params.schrodingerDisplayMode}, amp=${this.params.waveAmplitude}`);
        //}

        // Create bind group for visualization
        const vizBindGroup = this.device.createBindGroup({
            layout: this.schrodingerVisualizationBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiR[bufferIndex] } },
                { binding: 1, resource: { buffer: this.buffers.psiI[bufferIndex] } },
                { binding: 2, resource: this.textures.outputView },
                { binding: 3, resource: { buffer: this.schrodingerVisualizationParamsBuffer } }
            ]
        });

        // Run visualization compute pass
        const vizPass = commandEncoder.beginComputePass({
            label: 'Schrödinger visualization pass'
        });

        vizPass.setPipeline(this.schrodingerVisualizationPipeline);
        vizPass.setBindGroup(0, vizBindGroup);

        const workgroupsX = Math.ceil(this.width / 8);
        const workgroupsY = Math.ceil(this.height / 8);
        vizPass.dispatchWorkgroups(workgroupsX, workgroupsY);

        vizPass.end();
    }

    createSchrodingerVisualizationPipeline() {
        // Create visualization params buffer
        // Struct has: width(4) + height(4) + time(4) + amplitude_scale(4) + gamma(4) + display_mode(4) + prob_scale(4) + padding(4) = 32 bytes
        this.schrodingerVisualizationParamsBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Schrödinger visualization params'
        });

        const vizShaderCode = this.shaderManager.getSchrodingerVisualizationShader();

        const shaderModule = this.device.createShaderModule({
            code: vizShaderCode,
            label: 'Schrödinger visualization shader'
        });

        this.schrodingerVisualizationPipeline = this.device.createComputePipeline({
            label: 'Schrödinger visualization pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.schrodingerVisualizationBindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    createSchrodingerComputePipeline() {
        // Create a dedicated pipeline for Schrödinger computation
        const shaderCode = this.shaderManager.getSchrodingerShader();

        const shaderModule = this.device.createShaderModule({
            code: shaderCode,
            label: 'Schrödinger compute shader'
        });

        this.schrodingerComputePipeline = this.device.createComputePipeline({
            label: 'Schrödinger compute pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.schrodingerBindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });

        console.log('Created Schrödinger compute pipeline');
    }

    getCurrentTexture() {
        return this.textures.output;
    }

    async getBufferMinMax() {
        // Create a staging buffer to read back GPU data
        const bufferSize = this.width * this.height * 4; // 4 bytes per float
        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: 'Staging buffer for min/max calculation'
        });

        // Get the current buffer based on equation type with better error checking
        let sourceBuffer = null;

        try {
            if (this.currentEquationType === 'lattice-boltzmann' || this.currentEquationType === 'navier-stokes-fluid') {
                // For fluid simulations, use density buffer
                if (this.buffers && this.buffers.density && Array.isArray(this.buffers.density)) {
                    const bufferIndex = 1 - this.fluidBufferIndex;
                    sourceBuffer = this.buffers.density[bufferIndex];
                    console.log(`Using fluid density buffer [${bufferIndex}] for ${this.currentEquationType}`);
                }
            } else {
                // For wave simulations, use current buffer
                if (this.buffers && this.buffers.current) {
                    sourceBuffer = this.buffers.current;
                    console.log(`Using wave current buffer for ${this.currentEquationType}`);
                }
            }

            // Fallback: try to find any valid buffer
            if (!sourceBuffer) {
                console.warn('Primary buffer not found, trying fallbacks...');
                if (this.buffers) {
                    // Try different buffer types
                    if (this.buffers.current) {
                        sourceBuffer = this.buffers.current;
                        console.log('Using fallback: current buffer');
                    } else if (this.buffers.previous) {
                        sourceBuffer = this.buffers.previous;
                        console.log('Using fallback: previous buffer');
                    } else if (this.buffers.density && this.buffers.density[0]) {
                        sourceBuffer = this.buffers.density[0];
                        console.log('Using fallback: density buffer [0]');
                    }
                }
            }

            if (!sourceBuffer) {
                throw new Error(`No valid buffer found for equation type: ${this.currentEquationType}`);
            }

            // Verify it's actually a GPUBuffer
            if (!sourceBuffer || typeof sourceBuffer.copyBufferToBuffer === 'undefined') {
                throw new Error('Source buffer is not a valid GPUBuffer');
            }

            // Copy GPU buffer to staging buffer
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, bufferSize);
            this.device.queue.submit([commandEncoder.finish()]);

            // Map and read the staging buffer
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const arrayBuffer = stagingBuffer.getMappedRange();
            const data = new Float32Array(arrayBuffer);

            // Find min and max values
            let min = data[0];
            let max = data[0];

            for (let i = 1; i < data.length; i++) {
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }

            // Clean up
            stagingBuffer.unmap();
            stagingBuffer.destroy();

            console.log(`Buffer min/max analysis complete: min=${min.toFixed(3)}, max=${max.toFixed(3)}`);
            return { min, max };

        } catch (error) {
            // Clean up staging buffer on error
            try {
                if (stagingBuffer.mapState === 'mapped') {
                    stagingBuffer.unmap();
                }
                stagingBuffer.destroy();
            } catch (cleanupError) {
                console.warn('Error during cleanup:', cleanupError);
            }

            console.error('Buffer analysis failed:', error);
            console.log('Available buffers:', Object.keys(this.buffers || {}));
            console.log('Current equation type:', this.currentEquationType);
            console.log('Fluid buffer index:', this.fluidBufferIndex);

            throw error;
        }
    }

    setEquationType(equationType) {
        if (equationType !== this.currentEquationType) {
            console.log(`Changing equation type from ${this.currentEquationType} to ${equationType}`);
            this.currentEquationType = equationType;

            // Reset Schrödinger pipelines so they get recreated with correct layouts
            this.schrodingerComputePipeline = null;
            this.schrodingerVisualizationPipeline = null;

            this.recompileShaders().then(() => {
                if (equationType === 'lattice-boltzmann') {
                    console.log('LBM mode activated - reinitializing with proper initial conditions');
                    this.initializeWaveField();
                } else if (equationType === 'navier-stokes-fluid') {
                    console.log('Navier-Stokes mode activated - reinitializing fluid fields');
                    this.initializeWaveField();
                } else if (equationType === 'schrodinger') {
                    console.log('Schrödinger mode activated - reinitializing quantum wavefunction');
                    this.initializeWaveField();
                }
            }).catch(error => {
                console.error('Failed to recompile shaders for equation type:', equationType, error);
            });
        }
    }

    reset() {
        this.time = 0.0;
        this.params.time = 0.0;
        this.initializeWaveField();
        console.log('Wave simulation reset');
    }

    cleanup() {
        // Cleanup WebGPU resources
        for (let i = 0; i < 3; i++) {
            if (this.buffers.waveBuffers[i]) {
                this.buffers.waveBuffers[i].destroy();
            }
        }

        // Cleanup fluid buffers
        for (let i = 0; i < 2; i++) {
            if (this.buffers.velocityX[i]) this.buffers.velocityX[i].destroy();
            if (this.buffers.velocityY[i]) this.buffers.velocityY[i].destroy();
            if (this.buffers.density[i]) this.buffers.density[i].destroy();
        }

        if (this.buffers.boundary) this.buffers.boundary.destroy();
        if (this.buffers.params) this.buffers.params.destroy();
        if (this.buffers.webcamPrev) this.buffers.webcamPrev.destroy();
        if (this.buffers.webcamCurr) this.buffers.webcamCurr.destroy();
        if (this.buffers.pressure) this.buffers.pressure.destroy();
        if (this.buffers.divergence) this.buffers.divergence.destroy();

        // Cleanup Schrödinger buffers
        for (let i = 0; i < 2; i++) {
            if (this.buffers.psiR[i]) this.buffers.psiR[i].destroy();
            if (this.buffers.psiI[i]) this.buffers.psiI[i].destroy();
        }
        if (this.buffers.potential) this.buffers.potential.destroy();

        if (this.textures.output) this.textures.output.destroy();
    }
}
