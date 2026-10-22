export class FFTProcessor {
    constructor(device, width, height) {
        this.device = device;
        this.width = width;
        this.height = height;

        // FFT requires power-of-2 dimensions for efficiency
        this.fftWidth = this.nextPowerOf2(width);
        this.fftHeight = this.nextPowerOf2(height);

        // WebGPU resources
        this.buffers = {
            // Complex number buffers (real, imaginary pairs)
            webcamComplex: null,
            waveComplex: null,
            webcamFreq: null,
            waveFreq: null,
            resultFreq: null,
            resultComplex: null,

            // Intermediate buffers for FFT stages
            tempComplex1: null,
            tempComplex2: null,

            // Twiddle factors for FFT
            twiddleFactors: null
        };

        this.pipelines = {
            // FFT pipelines
            fftHorizontal: null,
            fftVertical: null,
            ifftHorizontal: null,
            ifftVertical: null,

            // Preprocessing pipelines
            realToComplex: null,
            complexToReal: null,

            // Convolution pipeline
            complexMultiply: null
        };

        this.bindGroupLayouts = {
            fft: null,
            conversion: null,
            multiply: null
        };
    }

    nextPowerOf2(n) {
        return Math.pow(2, Math.ceil(Math.log2(n)));
    }

    async initialize() {
        await this.createBuffers();
        await this.createBindGroupLayouts();
        await this.createShaders();
        await this.generateTwiddleFactors();

        console.log(`FFT Processor initialized for ${this.fftWidth}x${this.fftHeight}`);
    }

    async createBuffers() {
        const complexSize = this.fftWidth * this.fftHeight * 2 * 4; // 2 floats per complex number, 4 bytes per float

        // Create all complex buffers
        const bufferNames = ['webcamComplex', 'waveComplex', 'webcamFreq', 'waveFreq', 'resultFreq', 'resultComplex', 'tempComplex1', 'tempComplex2'];

        for (const name of bufferNames) {
            this.buffers[name] = this.device.createBuffer({
                size: complexSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                label: `FFT ${name} buffer`
            });
        }

        // Twiddle factors buffer (precomputed for efficiency)
        const twiddleSize = Math.max(this.fftWidth, this.fftHeight) * 2 * 4; // Complex numbers
        this.buffers.twiddleFactors = this.device.createBuffer({
            size: twiddleSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'FFT twiddle factors'
        });
    }

    async createBindGroupLayouts() {
        // FFT bind group layout (input, output, twiddle factors, params)
        this.bindGroupLayouts.fft = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // output
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // twiddle factors
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } // params
            ]
        });

        // Conversion bind group layout (real input, complex output)
        this.bindGroupLayouts.conversion = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // output
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } // params
            ]
        });

        // Complex multiplication bind group layout
        this.bindGroupLayouts.multiply = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input1
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input2
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // output
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } } // params
            ]
        });
    }

    async createShaders() {
        // Real to complex conversion shader
        const realToComplexShader = `
@group(0) @binding(0) var<storage, read> real_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> complex_output: array<f32>; // [real, imag, real, imag, ...]
@group(0) @binding(2) var<uniform> params: ConversionParams;

struct ConversionParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.output_width || y >= params.output_height) {
        return;
    }
    
    let output_idx = (y * params.output_width + x) * 2; // 2 floats per complex
    
    // Zero-pad if outside input dimensions
    if (x < params.input_width && y < params.input_height) {
        let input_idx = y * params.input_width + x;
        complex_output[output_idx] = real_input[input_idx]; // Real part
        complex_output[output_idx + 1] = 0.0; // Imaginary part
    } else {
        complex_output[output_idx] = 0.0; // Real part
        complex_output[output_idx + 1] = 0.0; // Imaginary part
    }
}`;

        // Complex to real conversion shader
        const complexToRealShader = `
@group(0) @binding(0) var<storage, read> complex_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> real_output: array<f32>;
@group(0) @binding(2) var<uniform> params: ConversionParams;

struct ConversionParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.output_width || y >= params.output_height) {
        return;
    }
    
    let input_idx = (y * params.input_width + x) * 2;
    let output_idx = y * params.output_width + x;
    
    // Extract magnitude (or just real part for simplicity)
    let real_part = complex_input[input_idx];
    let imag_part = complex_input[input_idx + 1];
    
    // Use magnitude for output
    real_output[output_idx] = sqrt(real_part * real_part + imag_part * imag_part);
}`;

        // 1D FFT shader (can be used for both horizontal and vertical passes)
        const fftShader = `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<storage, read> twiddle_factors: array<f32>;
@group(0) @binding(3) var<uniform> params: FFTParams;

struct FFTParams {
    width: u32,
    height: u32,
    stage: u32,
    is_horizontal: u32,
    is_inverse: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let N = select(params.height, params.width, params.is_horizontal == 1u);
    let stage_size = 1u << params.stage;
    let half_stage = stage_size >> 1u;
    
    var idx: u32;
    var partner_idx: u32;
    var twiddle_idx: u32;
    
    if (params.is_horizontal == 1u) {
        // Horizontal FFT
        let group = x / stage_size;
        let pos_in_group = x % stage_size;
        
        if (pos_in_group < half_stage) {
            idx = (y * params.width + group * stage_size + pos_in_group) * 2;
            partner_idx = (y * params.width + group * stage_size + pos_in_group + half_stage) * 2;
            twiddle_idx = pos_in_group * 2;
        } else {
            return; // Only process first half of each group
        }
    } else {
        // Vertical FFT
        let group = y / stage_size;
        let pos_in_group = y % stage_size;
        
        if (pos_in_group < half_stage) {
            idx = ((group * stage_size + pos_in_group) * params.width + x) * 2;
            partner_idx = ((group * stage_size + pos_in_group + half_stage) * params.width + x) * 2;
            twiddle_idx = pos_in_group * 2;
        } else {
            return; // Only process first half of each group
        }
    }
    
    // Load complex numbers
    let a_real = input_data[idx];
    let a_imag = input_data[idx + 1];
    let b_real = input_data[partner_idx];
    let b_imag = input_data[partner_idx + 1];
    
    // Load twiddle factor
    var w_real = twiddle_factors[twiddle_idx];
    var w_imag = twiddle_factors[twiddle_idx + 1];
    
    // Conjugate for inverse FFT
    if (params.is_inverse == 1u) {
        w_imag = -w_imag;
    }
    
    // Complex multiplication: b * w
    let bw_real = b_real * w_real - b_imag * w_imag;
    let bw_imag = b_real * w_imag + b_imag * w_real;
    
    // Butterfly operation
    let out_a_real = a_real + bw_real;
    let out_a_imag = a_imag + bw_imag;
    let out_b_real = a_real - bw_real;
    let out_b_imag = a_imag - bw_imag;
    
    // Store results
    output_data[idx] = out_a_real;
    output_data[idx + 1] = out_a_imag;
    output_data[partner_idx] = out_b_real;
    output_data[partner_idx + 1] = out_b_imag;
}`;

        // Complex multiplication shader (for frequency domain multiplication)
        const complexMultiplyShader = `
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MultiplyParams;

struct MultiplyParams {
    width: u32,
    height: u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = (y * params.width + x) * 2;
    
    // Load complex numbers
    let a_real = input1[idx];
    let a_imag = input1[idx + 1];
    let b_real = input2[idx];
    let b_imag = input2[idx + 1];
    
    // Complex multiplication: (a_real + i*a_imag) * (b_real + i*b_imag)
    let result_real = a_real * b_real - a_imag * b_imag;
    let result_imag = a_real * b_imag + a_imag * b_real;
    
    output[idx] = result_real;
    output[idx + 1] = result_imag;
}`;

        // Create shader modules
        this.shaders = {
            realToComplex: this.device.createShaderModule({ code: realToComplexShader }),
            complexToReal: this.device.createShaderModule({ code: complexToRealShader }),
            fft: this.device.createShaderModule({ code: fftShader }),
            complexMultiply: this.device.createShaderModule({ code: complexMultiplyShader })
        };

        // Create compute pipelines
        this.pipelines.realToComplex = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayouts.conversion] }),
            compute: { module: this.shaders.realToComplex, entryPoint: 'main' }
        });

        this.pipelines.complexToReal = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayouts.conversion] }),
            compute: { module: this.shaders.complexToReal, entryPoint: 'main' }
        });

        this.pipelines.fft = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayouts.fft] }),
            compute: { module: this.shaders.fft, entryPoint: 'main' }
        });

        this.pipelines.complexMultiply = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayouts.multiply] }),
            compute: { module: this.shaders.complexMultiply, entryPoint: 'main' }
        });
    }

    async generateTwiddleFactors() {
        const maxSize = Math.max(this.fftWidth, this.fftHeight);
        const twiddleFactors = new Float32Array(maxSize * 2);

        // Generate twiddle factors: e^(-2πik/N) for k = 0 to N/2-1
        for (let k = 0; k < maxSize / 2; k++) {
            const angle = -2.0 * Math.PI * k / maxSize;
            twiddleFactors[k * 2] = Math.cos(angle);     // Real part
            twiddleFactors[k * 2 + 1] = Math.sin(angle); // Imaginary part
        }

        this.device.queue.writeBuffer(this.buffers.twiddleFactors, 0, twiddleFactors);
    }

    // Main function to perform convoluted blending
    async performConvolutedBlending(webcamBuffer, waveBuffer, outputBuffer) {
        const commandEncoder = this.device.createCommandEncoder({ label: 'FFT Convolution' });

        // Step 1: Convert real data to complex
        await this.convertRealToComplex(commandEncoder, webcamBuffer, this.buffers.webcamComplex);
        await this.convertRealToComplex(commandEncoder, waveBuffer, this.buffers.waveComplex);

        // Step 2: Forward FFT on both arrays
        await this.performFFT(commandEncoder, this.buffers.webcamComplex, this.buffers.webcamFreq, false);
        await this.performFFT(commandEncoder, this.buffers.waveComplex, this.buffers.waveFreq, false);

        // Step 3: Element-wise multiplication in frequency domain
        await this.multiplyComplex(commandEncoder, this.buffers.webcamFreq, this.buffers.waveFreq, this.buffers.resultFreq);

        // Step 4: Inverse FFT
        await this.performFFT(commandEncoder, this.buffers.resultFreq, this.buffers.resultComplex, true);

        // Step 5: Convert complex result back to real
        await this.convertComplexToReal(commandEncoder, this.buffers.resultComplex, outputBuffer);

        // Submit all commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    async convertRealToComplex(commandEncoder, inputBuffer, outputBuffer) {
        // Create parameters buffer
        const params = new ArrayBuffer(16);
        const view = new DataView(params);
        view.setUint32(0, this.width, true);
        view.setUint32(4, this.height, true);
        view.setUint32(8, this.fftWidth, true);
        view.setUint32(12, this.fftHeight, true);

        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, params);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.conversion,
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: paramsBuffer } }
            ]
        });

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.pipelines.realToComplex);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.fftWidth / 8),
            Math.ceil(this.fftHeight / 8)
        );
        computePass.end();

        paramsBuffer.destroy();
    }

    async convertComplexToReal(commandEncoder, inputBuffer, outputBuffer) {
        // Create parameters buffer
        const params = new ArrayBuffer(16);
        const view = new DataView(params);
        view.setUint32(0, this.fftWidth, true);
        view.setUint32(4, this.fftHeight, true);
        view.setUint32(8, this.width, true);
        view.setUint32(12, this.height, true);

        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, params);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.conversion,
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: paramsBuffer } }
            ]
        });

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.pipelines.complexToReal);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.width / 8),
            Math.ceil(this.height / 8)
        );
        computePass.end();

        paramsBuffer.destroy();
    }

    async performFFT(commandEncoder, inputBuffer, outputBuffer, isInverse) {
        const size = Math.max(this.fftWidth, this.fftHeight);
        const stages = Math.log2(size);

        let currentInput = inputBuffer;
        let currentOutput = outputBuffer;

        // Horizontal FFT passes
        for (let stage = 0; stage < Math.log2(this.fftWidth); stage++) {
            await this.performFFTStage(commandEncoder, currentInput, currentOutput, stage, true, isInverse);
            [currentInput, currentOutput] = [currentOutput, currentInput];
        }

        // Vertical FFT passes
        for (let stage = 0; stage < Math.log2(this.fftHeight); stage++) {
            await this.performFFTStage(commandEncoder, currentInput, currentOutput, stage, false, isInverse);
            [currentInput, currentOutput] = [currentOutput, currentInput];
        }

        // Ensure final result is in the correct output buffer
        if (currentInput !== outputBuffer) {
            commandEncoder.copyBufferToBuffer(
                currentInput, 0,
                outputBuffer, 0,
                this.fftWidth * this.fftHeight * 2 * 4
            );
        }

        // Normalization for inverse FFT
        if (isInverse) {
            // TODO: Add normalization pass to divide by N
        }
    }

    async performFFTStage(commandEncoder, inputBuffer, outputBuffer, stage, isHorizontal, isInverse) {
        const params = new ArrayBuffer(20);
        const view = new DataView(params);
        view.setUint32(0, this.fftWidth, true);
        view.setUint32(4, this.fftHeight, true);
        view.setUint32(8, stage, true);
        view.setUint32(12, isHorizontal ? 1 : 0, true);
        view.setUint32(16, isInverse ? 1 : 0, true);

        const paramsBuffer = this.device.createBuffer({
            size: 20,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, params);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.fft,
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: this.buffers.twiddleFactors } },
                { binding: 3, resource: { buffer: paramsBuffer } }
            ]
        });

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.pipelines.fft);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.fftWidth / 8),
            Math.ceil(this.fftHeight / 8)
        );
        computePass.end();

        paramsBuffer.destroy();
    }

    async multiplyComplex(commandEncoder, input1Buffer, input2Buffer, outputBuffer) {
        const params = new ArrayBuffer(8);
        const view = new DataView(params);
        view.setUint32(0, this.fftWidth, true);
        view.setUint32(4, this.fftHeight, true);

        const paramsBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(paramsBuffer, 0, params);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayouts.multiply,
            entries: [
                { binding: 0, resource: { buffer: input1Buffer } },
                { binding: 1, resource: { buffer: input2Buffer } },
                { binding: 2, resource: { buffer: outputBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } }
            ]
        });

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.pipelines.complexMultiply);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.fftWidth / 8),
            Math.ceil(this.fftHeight / 8)
        );
        computePass.end();

        paramsBuffer.destroy();
    }

    cleanup() {
        // Destroy all buffers
        Object.values(this.buffers).forEach(buffer => {
            if (buffer) buffer.destroy();
        });
    }
}

