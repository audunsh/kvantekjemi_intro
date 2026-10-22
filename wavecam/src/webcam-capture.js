export class WebcamCapture {
    constructor() {
        this.video = null;
        this.stream = null;
        this.width = 320;
        this.height = 240;
        this.mirror = false;
        this.texture = null;
    }

    async initialize() {
        // Get video element
        this.video = document.getElementById('webcam');
        if (!this.video) {
            throw new Error('Webcam video element not found');
        }

        try {
            // Request webcam access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: this.width },
                    height: { ideal: this.height },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.muted = true;
            this.video.playsInline = true;

            const ready = new Promise((resolve, reject) => {
                if (this.video.readyState >= 1) {
                    resolve();
                    return;
                }
                this.video.addEventListener('loadedmetadata', () => resolve(), { once: true });
                this.video.addEventListener('error', () => reject(new Error('Webcam video element error')), { once: true });
            });
            const timeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Webcam initialization timeout')), 10000);
            });

            this.video.srcObject = this.stream;
            await Promise.race([
                Promise.all([this.video.play(), ready]),
                timeout
            ]);

            // Update dimensions based on actual video
            this.width = this.video.videoWidth || this.width;
            this.height = this.video.videoHeight || this.height;

            console.log(`Webcam initialized: ${this.width}x${this.height}`);

        } catch (error) {
            if (this.stream) {
                this.stream.getTracks().forEach((track) => track.stop());
                this.stream = null;
            }
            if (this.video) {
                this.video.srcObject = null;
            }
            throw new Error(`Failed to initialize webcam: ${error.message}`);
        }
    }

    updateTexture(device) {
        if (!this.video || this.video.readyState < 2) {
            return null;
        }

        // Create or recreate texture if needed
        if (!this.texture) {
            this.texture = device.createTexture({
                size: [this.width, this.height, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT
            });
        }

        // Create a canvas to extract pixel data
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');

        // Draw video frame to canvas
        ctx.drawImage(this.video, 0, 0, this.width, this.height);

        // Get image data
        const imageData = ctx.getImageData(0, 0, this.width, this.height);

        // Write to WebGPU texture
        device.queue.writeTexture(
            { texture: this.texture },
            imageData.data,
            { bytesPerRow: this.width * 4 },
            { width: this.width, height: this.height }
        );

        return this.texture;
    }

    getGrayscaleTexture(device) {
        if (!this.video || this.video.readyState < 2) {
            return null;
        }

        // Create canvas for grayscale conversion
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');

        // Draw video frame
        ctx.drawImage(this.video, 0, 0, this.width, this.height);

        // Get image data and convert to grayscale
        const imageData = ctx.getImageData(0, 0, this.width, this.height);
        const data = imageData.data;

        // Convert to grayscale using luminance formula
        for (let i = 0; i < data.length; i += 4) {
            const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
            data[i] = gray;     // R
            data[i + 1] = gray; // G
            data[i + 2] = gray; // B
            // Alpha remains unchanged
        }

        // Create texture if it doesn't exist
        if (!this.grayscaleTexture) {
            this.grayscaleTexture = device.createTexture({
                size: [this.width, this.height, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST
            });
        }

        // Write grayscale data to texture
        device.queue.writeTexture(
            { texture: this.grayscaleTexture },
            imageData.data,
            { bytesPerRow: this.width * 4 },
            { width: this.width, height: this.height }
        );

        return this.grayscaleTexture;
    }

    getTextureSize() {
        return { width: this.width, height: this.height };
    }

    getBoundaryMask(threshold = 0.4) {
        if (!this.video || this.video.readyState < 2) {
            return null;
        }

        // Create canvas for boundary detection
        const canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');

        // Draw video frame
        ctx.drawImage(this.video, 0, 0, this.width, this.height);

        // Get image data
        const imageData = ctx.getImageData(0, 0, this.width, this.height);
        const data = imageData.data;

        // Create boundary mask (1.0 for boundary, 0.0 for free space)
        const boundaryData = new Float32Array(this.width * this.height);

        for (let i = 0; i < data.length; i += 4) {
            // Calculate grayscale value (0-1 range)
            const gray = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255.0;
            const pixelIndex = i / 4;

            // Set boundary if pixel is darker than threshold
            boundaryData[pixelIndex] = gray < threshold ? 1.0 : 0.0;
        }

        return boundaryData;
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.video) {
            this.video.srcObject = null;
        }

        if (this.texture) {
            this.texture.destroy();
            this.texture = null;
        }

        if (this.grayscaleTexture) {
            this.grayscaleTexture.destroy();
            this.grayscaleTexture = null;
        }
    }

    setMirror(mirror) {
        this.mirror = mirror;
    }
}
