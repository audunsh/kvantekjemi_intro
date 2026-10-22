export class WebGPUCore {
    constructor() {
        this.adapter = null;
        this.device = null;
        this.context = null;
        this.canvas = null;
        this.preferredFormat = null;
    }

    async initialize() {
        // Get the canvas
        this.canvas = document.getElementById('simulation-canvas');
        if (!this.canvas) {
            throw new Error('Canvas element not found');
        }

        // Request WebGPU adapter
        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!this.adapter) {
            throw new Error('Failed to get WebGPU adapter');
        }

        // Request device
        this.device = await this.adapter.requestDevice({
            // Avoid strict requiredLimits for wider compatibility
        });

        // Set up error handling
        this.device.addEventListener('uncapturederror', (event) => {
            console.error('WebGPU uncaptured error:', event.error);
        });

        // Get the canvas context
        this.context = this.canvas.getContext('webgpu');
        if (!this.context) {
            throw new Error('Failed to get WebGPU context');
        }

        // Get preferred canvas format
        this.preferredFormat = navigator.gpu.getPreferredCanvasFormat();

        // Configure the canvas
        this.context.configure({
            device: this.device,
            format: this.preferredFormat,
            alphaMode: 'premultiplied'
        });

        console.log('WebGPU initialized successfully');

        // requestAdapterInfo is not available in all implementations yet
        try {
            if (this.adapter.requestAdapterInfo) {
                const adapterInfo = await this.adapter.requestAdapterInfo();
                console.log('Adapter info:', adapterInfo);
            }
        } catch (error) {
            console.log('Adapter info not available');
        }

        console.log('Device limits:', this.device.limits);
    }

    createBuffer(size, usage, mappedAtCreation = false) {
        return this.device.createBuffer({
            size,
            usage,
            mappedAtCreation
        });
    }

    createTexture(descriptor) {
        return this.device.createTexture(descriptor);
    }

    createBindGroup(descriptor) {
        return this.device.createBindGroup(descriptor);
    }

    createBindGroupLayout(descriptor) {
        return this.device.createBindGroupLayout(descriptor);
    }

    createComputePipeline(descriptor) {
        return this.device.createComputePipeline(descriptor);
    }

    createRenderPipeline(descriptor) {
        return this.device.createRenderPipeline(descriptor);
    }

    createShaderModule(descriptor) {
        return this.device.createShaderModule(descriptor);
    }

    createCommandEncoder() {
        return this.device.createCommandEncoder();
    }

    submitCommands(commandBuffer) {
        this.device.queue.submit([commandBuffer]);
    }

    writeBuffer(buffer, data, offset = 0) {
        this.device.queue.writeBuffer(buffer, offset, data);
    }

    writeTexture(destination, data, dataLayout, size) {
        this.device.queue.writeTexture(destination, data, dataLayout, size);
    }

    getCurrentTexture() {
        return this.context.getCurrentTexture();
    }

    getPreferredFormat() {
        return this.preferredFormat;
    }
}
