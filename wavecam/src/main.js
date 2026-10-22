import { WebGPUCore } from './webgpu-core.js';
import { WebcamCapture } from './webcam-capture.js';
import { WaveSimulation } from './wave-simulation.js';
import { RenderPipeline } from './render-pipeline.js';
import { ShaderManager } from './shader-manager.js';

class WaveCamApp {
    constructor() {
        this.webgpuCore = null;
        this.webcamCapture = null;
        this.waveSimulation = null;
        this.renderPipeline = null;
        this.shaderManager = null;

        this.isRunning = false;
        this.animationId = null;

        // Simulation parameters
        this.params = {
            waveSpeed: 2.0,
            damping: 0.0,
            sourceFrequency: 0.1,
            boundaryThreshold: 0.25,
            sourceEnabled: true,
            equationType: 'wave',
            waveAmplitude: 10.0,
            sourceStrength: 2.0,
            sourceSize: 5.0,
            normalizeWaves: true,
            waveRange: 1.0,
            stepsPerFrame: 3,
            mixRatio: 0.7,
            colorScheme: 'red-blue',
            blendMode: 'normal',

            // Cloth physics parameters
            clothGravity: 0.5,
            clothStiffness: 0.8,
            clothDamping: 0.95,
            motionSensitivity: 2.0,

            // Boundary inversion
            invertBoundaries: false,

            // Lattice Boltzmann and Navier-Stokes parameters
            fluidViscosity: 0.2,     // Higher base viscosity (shader scales it down for turbulence)
            flowVelocity: 0.05,
            motionForce: 1.0,        // Motion force multiplier
            densityDisplay: 2.0,     // Increased for better visibility

            // Fluid flow direction and strength
            flowDirection: 0.0, // degrees
            flowStrength: 0.0,  // No initial flow - user can control it

            // Color scale parameters - wider default range to capture more wave dynamics
            colorScaleMin: -5.0,
            colorScaleMax: 5.0,

            // Normal mapping parameters
            enableNormalMapping: false,
            normalStrength: 1.0,
            lightDirection: 45.0,  // degrees
            lightElevation: 30.0,  // degrees

            // Chromatic aberration parameters
            enableChromaticAberration: false,
            aberrationStrength: 2.0,
            aberrationOffset: 0.5,
            aberrationAngle: 0.0,   // degrees

            // Schrödinger visualization parameters
            schrodingerDisplayMode: 0.0,  // 0=phase colors, 1=probability, 2=phase wheel
            schrodingerProbScale: 500.0,  // Scale factor for probability density
            schrodingerSigma: 30.0,       // Initial Gaussian width (localization)
            schrodingerKx: 0.8,           // Initial x-momentum
            schrodingerKy: 0.0,           // Initial y-momentum
            schrodingerVOffset: 0.0       // Potential offset for V(x,y)
        };

        // Preset management (in-memory for current session)
        this.presets = [];

        this.initializeUI();
    }

    async initialize() {
        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                throw new Error('WebGPU is not supported in this browser. Please use Chrome Canary or another WebGPU-enabled browser.');
            }

            // Initialize WebGPU core
            this.webgpuCore = new WebGPUCore();
            await this.webgpuCore.initialize();

            // Initialize webcam capture
            this.webcamCapture = new WebcamCapture();
            await this.webcamCapture.initialize();

            // Initialize shader manager
            this.shaderManager = new ShaderManager();

            // Initialize wave simulation
            this.waveSimulation = new WaveSimulation(
                this.webgpuCore.device,
                this.webcamCapture.getTextureSize(),
                this.shaderManager
            );
            await this.waveSimulation.initialize();

            // Initialize render pipeline
            this.renderPipeline = new RenderPipeline(
                this.webgpuCore.device,
                this.webgpuCore.context,
                this.webcamCapture.getTextureSize()
            );
            await this.renderPipeline.initialize();

            // Start the simulation loop
            this.start();

            console.log('WaveCam application initialized successfully');

        } catch (error) {
            this.showError(error.message);
            console.error('Failed to initialize WaveCam:', error);
        }
    }

    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    animate() {
        if (!this.isRunning) return;

        try {
            // Update webcam texture (once per frame)
            const webcamTexture = this.webcamCapture.updateTexture(this.webgpuCore.device);

            // Update simulation parameters
            this.waveSimulation.updateParameters(this.params);

            // Run multiple simulation steps per frame for higher temporal resolution
            const stepsPerFrame = Math.max(1, Math.min(20, this.params.stepsPerFrame || 3));

            for (let step = 0; step < stepsPerFrame; step++) {
                // Only update webcam boundaries on the first step to avoid redundant work
                this.waveSimulation.step(step === 0 ? webcamTexture : null);
            }

            // Render the result (once per frame)
            this.renderPipeline.render(
                this.waveSimulation.getCurrentTexture(),
                webcamTexture,
                {
                    mixRatio: this.params.mixRatio,
                    colorScheme: this.params.colorScheme,
                    blendMode: this.params.blendMode,
                    time: this.waveSimulation.time,
                    colorScaleMin: this.params.colorScaleMin,
                    colorScaleMax: this.params.colorScaleMax,
                    enableNormalMapping: this.params.enableNormalMapping,
                    normalStrength: this.params.normalStrength,
                    lightDirection: this.params.lightDirection,
                    lightElevation: this.params.lightElevation,
                    enableChromaticAberration: this.params.enableChromaticAberration,
                    aberrationStrength: this.params.aberrationStrength,
                    aberrationOffset: this.params.aberrationOffset,
                    aberrationAngle: this.params.aberrationAngle
                }
            );

        } catch (error) {
            console.error('Animation loop error:', error);
            this.showError('Simulation error: ' + error.message);

            // Try to continue instead of stopping completely
            console.log('Attempting to continue simulation...');
        }

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    reset() {
        if (this.waveSimulation) {
            this.waveSimulation.reset();
        }
    }

    async autoScaleColors() {
        if (!this.waveSimulation) {
            console.warn('Wave simulation not initialized');
            this.showError('Wave simulation not initialized');
            return;
        }

        try {
            // Disable the button temporarily to prevent multiple clicks
            const button = document.getElementById('auto-scale-colors');
            if (button) {
                button.disabled = true;
                button.textContent = '⏳ Analyzing...';
            }

            // Get the current min/max values from the simulation buffer
            const { min, max } = await this.waveSimulation.getBufferMinMax();

            // Validate the results
            if (!isFinite(min) || !isFinite(max) || min === max) {
                throw new Error(`Invalid buffer data: min=${min}, max=${max}`);
            }

            // Update parameters with some padding to avoid edge clipping
            const range = max - min;
            const padding = range * 0.1; // 10% padding

            this.params.colorScaleMin = min - padding;
            this.params.colorScaleMax = max + padding;

            // Update UI sliders
            const minSlider = document.getElementById('color-scale-min');
            const maxSlider = document.getElementById('color-scale-max');
            const minValue = document.getElementById('color-scale-min-value');
            const maxValue = document.getElementById('color-scale-max-value');

            if (minSlider && maxSlider && minValue && maxValue) {
                // Ensure sliders can handle the new range
                const sliderMin = Math.min(-10, this.params.colorScaleMin - 1);
                const sliderMax = Math.max(10, this.params.colorScaleMax + 1);

                minSlider.min = sliderMin;
                minSlider.max = sliderMax;
                maxSlider.min = sliderMin;
                maxSlider.max = sliderMax;

                minSlider.value = this.params.colorScaleMin;
                maxSlider.value = this.params.colorScaleMax;
                minValue.textContent = this.params.colorScaleMin.toFixed(2);
                maxValue.textContent = this.params.colorScaleMax.toFixed(2);
            }

            console.log(`Auto-scaled colors: min=${this.params.colorScaleMin.toFixed(2)}, max=${this.params.colorScaleMax.toFixed(2)}`);

            // Re-enable the button
            if (button) {
                button.disabled = false;
                button.textContent = '🎯 Auto-Scale Colors';
            }

        } catch (error) {
            console.error('Failed to auto-scale colors:', error);
            this.showError(`Auto-scale failed: ${error.message}`);

            // Re-enable the button on error
            const button = document.getElementById('auto-scale-colors');
            if (button) {
                button.disabled = false;
                button.textContent = '🎯 Auto-Scale Colors';
            }
        }
    }

    toggleWebcamMirror(mirror) {
        const webcamVideo = document.getElementById('webcam');
        if (webcamVideo) {
            if (mirror) {
                webcamVideo.style.transform = 'scaleX(-1)';
            } else {
                webcamVideo.style.transform = 'scaleX(1)';
            }
        }

        // Store the mirror state for the webcam capture to use in texture processing
        if (this.webcamCapture) {
            this.webcamCapture.setMirror(mirror);
        }
    }

    async changeResolution(width, height) {
        try {
            // Update canvas sizes
            const simulationCanvas = document.getElementById('simulation-canvas');
            const waveOnlyCanvas = document.getElementById('wave-only-canvas');

            simulationCanvas.width = width;
            simulationCanvas.height = height;
            waveOnlyCanvas.width = width;
            waveOnlyCanvas.height = height;

            // Update resolution info display
            const resolutionInfo = document.getElementById('resolution-info');
            if (resolutionInfo) {
                resolutionInfo.textContent = `Resolution: ${width}×${height}`;
            }

            // Reinitialize simulation with new dimensions
            if (this.waveSimulation) {
                this.waveSimulation.cleanup();
            }
            if (this.renderPipeline) {
                this.renderPipeline.cleanup();
            }

            // Reinitialize with new size
            await this.initialize();

            console.log(`Resolution changed to ${width}×${height}`);
        } catch (error) {
            console.error('Failed to change resolution:', error);
            this.showError('Failed to change resolution: ' + error.message);
        }
    }

    initializeUI() {
        // Parameter controls
        this.setupSlider('wave-speed', 'waveSpeed');
        this.setupSlider('damping', 'damping');
        this.setupSlider('source-frequency', 'sourceFrequency');
        this.setupSlider('boundary-threshold', 'boundaryThreshold');
        this.setupSlider('wave-amplitude', 'waveAmplitude');
        this.setupSlider('source-strength', 'sourceStrength');
        this.setupSlider('source-size', 'sourceSize');
        this.setupSlider('wave-range', 'waveRange');
        this.setupSlider('steps-per-frame', 'stepsPerFrame');
        this.setupSlider('mix-ratio', 'mixRatio');

        // Cloth physics parameters (also used by LBM)
        this.setupSlider('cloth-gravity', 'clothGravity');       // LBM: Fluid Viscosity
        this.setupSlider('cloth-stiffness', 'clothStiffness');   // LBM: Flow Velocity  
        this.setupSlider('cloth-damping', 'clothDamping');       // LBM: Motion Force
        this.setupSlider('motion-sensitivity', 'motionSensitivity'); // LBM: Density Display

        // LBM sliders (use dedicated LBM parameters)
        this.setupSlider('fluid-viscosity', 'fluidViscosity');
        this.setupSlider('flow-velocity', 'flowVelocity');
        this.setupSlider('motion-force', 'motionForce');
        this.setupSlider('density-display', 'densityDisplay');

        // Navier-Stokes fluid sliders (dedicated NS parameters)
        this.setupSlider('ns-fluid-viscosity', 'fluidViscosity');
        this.setupSlider('ns-flow-velocity', 'flowVelocity');
        this.setupSlider('ns-motion-force', 'motionForce');
        this.setupSlider('ns-density-display', 'densityDisplay');
        this.setupSlider('ns-velocity-damping', 'damping');
        this.setupSlider('ns-density-damping', 'clothDamping');
        this.setupSlider('ns-flow-direction', 'flowDirection');
        this.setupSlider('ns-flow-strength', 'flowStrength');
        this.setupSlider('color-scale-min', 'colorScaleMin');
        this.setupSlider('color-scale-max', 'colorScaleMax');
        this.setupSlider('normal-strength', 'normalStrength');
        this.setupSlider('light-direction', 'lightDirection');
        this.setupSlider('light-elevation', 'lightElevation');
        this.setupSlider('aberration-strength', 'aberrationStrength');
        this.setupSlider('aberration-offset', 'aberrationOffset');
        this.setupSlider('aberration-angle', 'aberrationAngle');

        // Schrödinger visualization controls
        this.setupSlider('schrodinger-prob-scale', 'schrodingerProbScale');
        this.setupSlider('schrodinger-sigma', 'schrodingerSigma');
        this.setupSlider('schrodinger-kx', 'schrodingerKx');
        this.setupSlider('schrodinger-ky', 'schrodingerKy');
        this.setupSlider('schrodinger-v-offset', 'schrodingerVOffset');

        // Schrödinger display mode dropdown
        const schrodingerModeSelect = document.getElementById('schrodinger-display-mode');
        if (schrodingerModeSelect) {
            schrodingerModeSelect.addEventListener('change', (e) => {
                this.params.schrodingerDisplayMode = parseFloat(e.target.value);

                // Update hint text
                const hint = document.getElementById('schrodinger-mode-hint');
                if (hint) {
                    if (this.params.schrodingerDisplayMode > 2.5) {
                        hint.textContent = 'Color mode: ignores Color Scheme setting';
                    } else {
                        hint.textContent = 'Scalar modes use Color Scheme below';
                    }
                }

                // Update legend text based on mode
                const legend = document.getElementById('schrodinger-legend');
                if (legend) {
                    if (this.params.schrodingerDisplayMode < 0.5) {
                        legend.innerHTML = '<strong>Real Part ψ_R</strong><br>• Use Red-Blue colormap<br>• Positive (red) / Negative (blue)<br><span style="color: #aaa;">Shows wave oscillations</span>';
                    } else if (this.params.schrodingerDisplayMode < 1.5) {
                        legend.innerHTML = '<strong>Probability Density |ψ|²</strong><br>• Use Viridis/Inferno/Magma<br>• Higher = more likely location<br><span style="color: #aaa;">Born interpretation</span>';
                    } else if (this.params.schrodingerDisplayMode < 2.5) {
                        legend.innerHTML = '<strong>Phase-Weighted Magnitude</strong><br>• Use Red-Blue colormap<br>• Encodes phase + amplitude<br><span style="color: #aaa;">Full quantum information</span>';
                    } else if (this.params.schrodingerDisplayMode < 3.5) {
                        legend.innerHTML = '<strong>Real + Imaginary Combined</strong><br>🔴 Red = +Re &nbsp; 🔵 Blue = -Re<br>🟡 Yellow = +Im &nbsp; 🟢 Cyan = -Im<br><span style="color: #aaa;">Color = phase, Brightness = |ψ| amplitude<br>Rapid color changes = short wavelength (high p)</span>';
                    } else {
                        legend.innerHTML = '<strong>Imaginary Part ψ_I (Debug)</strong><br>• Use Red-Blue colormap<br>• Check if imaginary part exists<br><span style="color: #aaa;">Debug visualization</span>';
                    }
                }
            });
        }

        // Schrödinger reset button
        const resetSchrodingerBtn = document.getElementById('reset-schrodinger');
        if (resetSchrodingerBtn) {
            resetSchrodingerBtn.addEventListener('click', () => {
                if (this.waveSimulation && this.waveSimulation.currentEquationType === 'schrodinger') {
                    console.log('Resetting Schrödinger wavefunction with current parameters');
                    this.waveSimulation.initializeWaveField();
                }
            });
        }

        // Preset save button
        const savePresetBtn = document.getElementById('save-preset-btn');
        if (savePresetBtn) {
            savePresetBtn.addEventListener('click', () => this.savePreset());
        }

        // Allow Enter key in preset name input to save
        const presetNameInput = document.getElementById('preset-name');
        if (presetNameInput) {
            presetNameInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.savePreset();
                }
            });
        }

        // Initialize all slider displays with current parameter values
        this.initializeSliderDisplays();

        // Checkbox control for normalization
        document.getElementById('normalize-waves').addEventListener('change', (e) => {
            this.params.normalizeWaves = e.target.checked;
        });

        // Checkbox control for boundary inversion
        document.getElementById('invert-boundaries').addEventListener('change', (e) => {
            this.params.invertBoundaries = e.target.checked;
        });

        // Color scheme selector
        document.getElementById('color-scheme').addEventListener('change', (e) => {
            this.params.colorScheme = e.target.value;
        });

        // Blend mode selector
        document.getElementById('blend-mode').addEventListener('change', (e) => {
            this.params.blendMode = e.target.value;
        });

        // Equation type selector
        document.getElementById('equation-type').addEventListener('change', (e) => {
            this.params.equationType = e.target.value;
            console.log(`🔄 EQUATION TYPE CHANGED TO: ${e.target.value}`);
            if (this.waveSimulation) {
                this.waveSimulation.setEquationType(e.target.value);
                console.log(`📊 Current parameters:`, {
                    motionForce: this.params.motionForce,
                    sourceStrength: this.params.sourceStrength,
                    waveSpeed: this.params.waveSpeed,
                    damping: this.params.damping
                });
            }
            // Update equation info display
            const equationInfo = document.getElementById('equation-info');
            if (equationInfo) {
                const optionText = e.target.options[e.target.selectedIndex].text;
                equationInfo.textContent = optionText.split(' (')[0]; // Get just the main name
            }
        });

        // Resolution selector
        const resolutionSelect = document.getElementById('resolution');
        resolutionSelect.addEventListener('change', (e) => {
            const [width, height] = e.target.value.split('x').map(Number);
            this.changeResolution(width, height);
        });

        // Control buttons
        document.getElementById('reset-simulation').addEventListener('click', () => {
            this.reset();
        });

        document.getElementById('toggle-source').addEventListener('click', () => {
            this.params.sourceEnabled = !this.params.sourceEnabled;
            const button = document.getElementById('toggle-source');
            button.textContent = this.params.sourceEnabled ? 'Disable Source' : 'Enable Source';
        });

        // Auto-scale colors button
        document.getElementById('auto-scale-colors').addEventListener('click', () => {
            this.autoScaleColors();
        });

        // Mirror webcam checkbox
        document.getElementById('mirror-webcam').addEventListener('change', (e) => {
            this.toggleWebcamMirror(e.target.checked);
        });

        // Normal mapping checkbox
        document.getElementById('enable-normal-mapping').addEventListener('change', (e) => {
            this.params.enableNormalMapping = e.target.checked;
        });

        // Chromatic aberration checkbox
        document.getElementById('enable-chromatic-aberration').addEventListener('change', (e) => {
            this.params.enableChromaticAberration = e.target.checked;
        });

        document.getElementById('debug-mode').addEventListener('click', () => {
            if (this.waveSimulation) {
                this.waveSimulation.debugMode = !this.waveSimulation.debugMode;
                const button = document.getElementById('debug-mode');
                button.textContent = this.waveSimulation.debugMode ? 'Debug Mode: ON' : 'Debug Mode: OFF';
                console.log('Debug mode:', this.waveSimulation.debugMode ? 'ON' : 'OFF');
            }
        });

        // Shader editor controls
        document.getElementById('compile-shader').addEventListener('click', () => {
            this.compileCustomShader();
        });

        document.getElementById('load-default').addEventListener('click', () => {
            this.loadDefaultShader();
        });
    }

    setupSlider(elementId, paramName) {
        const slider = document.getElementById(elementId);
        const valueDisplay = document.getElementById(elementId + '-value');

        if (!slider || !valueDisplay) {
            console.warn(`Slider or display not found for ${elementId}`);
            return;
        }

        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.params[paramName] = value;

            // Map Schrödinger V offset to flowStrength for shader compatibility
            if (paramName === 'schrodingerVOffset') {
                this.params.flowStrength = value;
            }

            // Debug logging for Navier-Stokes controls
            if (elementId.startsWith('ns-')) {
                console.log(`NS Control: ${elementId} -> ${paramName} = ${value}`);
            }

            // Special formatting for flow direction
            if (elementId === 'ns-flow-direction') {
                valueDisplay.textContent = value.toFixed(0) + '°';
            } else {
                valueDisplay.textContent = value.toFixed(2);
            }
        });
    }

    initializeSliderDisplays() {
        // Initialize all slider displays with current parameter values
        const sliderMappings = [
            ['wave-speed', 'waveSpeed'],
            ['damping', 'damping'],
            ['source-frequency', 'sourceFrequency'],
            ['boundary-threshold', 'boundaryThreshold'],
            ['wave-amplitude', 'waveAmplitude'],
            ['source-strength', 'sourceStrength'],
            ['source-size', 'sourceSize'],
            ['wave-range', 'waveRange'],
            ['steps-per-frame', 'stepsPerFrame'],
            ['mix-ratio', 'mixRatio'],
            ['cloth-gravity', 'clothGravity'],
            ['cloth-stiffness', 'clothStiffness'],
            ['cloth-damping', 'clothDamping'],
            ['motion-sensitivity', 'motionSensitivity'],
            ['fluid-viscosity', 'fluidViscosity'],
            ['flow-velocity', 'flowVelocity'],
            ['motion-force', 'motionForce'],
            ['density-display', 'densityDisplay'],
            ['ns-fluid-viscosity', 'fluidViscosity'],
            ['ns-flow-velocity', 'flowVelocity'],
            ['ns-motion-force', 'motionForce'],
            ['ns-density-display', 'densityDisplay'],
            ['ns-velocity-damping', 'damping'],
            ['ns-density-damping', 'clothDamping'],
            ['ns-flow-direction', 'flowDirection'],
            ['ns-flow-strength', 'flowStrength']
        ];

        sliderMappings.forEach(([elementId, paramName]) => {
            const slider = document.getElementById(elementId);
            const valueDisplay = document.getElementById(elementId + '-value');

            if (slider && valueDisplay && this.params[paramName] !== undefined) {
                // Set slider value to match parameter
                slider.value = this.params[paramName];

                // Update display
                if (elementId === 'ns-flow-direction') {
                    valueDisplay.textContent = this.params[paramName].toFixed(0) + '°';
                } else {
                    valueDisplay.textContent = this.params[paramName].toFixed(2);
                }
            }
        });
    }

    compileCustomShader() {
        const shaderCode = document.getElementById('shader-editor').value;
        if (!shaderCode.trim()) {
            this.showError('Please enter shader code before compiling');
            return;
        }

        try {
            this.shaderManager.setCustomShader(shaderCode);
            if (this.waveSimulation) {
                this.waveSimulation.recompileShaders();
            }
            console.log('Custom shader compiled successfully');
        } catch (error) {
            this.showError('Shader compilation error: ' + error.message);
        }
    }

    loadDefaultShader() {
        const defaultShader = this.shaderManager.getDefaultWaveShader();
        document.getElementById('shader-editor').value = defaultShader;
    }

    showError(message) {
        const errorElement = document.getElementById('error-message');
        errorElement.textContent = message;
        errorElement.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    }

    // Preset Management Methods
    savePreset() {
        const presetNameInput = document.getElementById('preset-name');
        const presetName = presetNameInput.value.trim();

        if (!presetName) {
            this.showError('Please enter a preset name');
            return;
        }

        // Create a snapshot of current state
        const preset = {
            name: presetName,
            timestamp: Date.now(),
            params: JSON.parse(JSON.stringify(this.params)) // Deep clone
        };

        // Check if preset with same name exists
        const existingIndex = this.presets.findIndex(p => p.name === presetName);
        if (existingIndex >= 0) {
            // Update existing preset
            this.presets[existingIndex] = preset;
            console.log(`Updated preset: ${presetName}`);
        } else {
            // Add new preset
            this.presets.push(preset);
            console.log(`Saved new preset: ${presetName}`);
        }

        // Clear input and update UI
        presetNameInput.value = '';
        this.updatePresetList();
    }

    loadPreset(index) {
        if (index < 0 || index >= this.presets.length) {
            this.showError('Invalid preset');
            return;
        }

        const preset = this.presets[index];
        console.log(`Loading preset: ${preset.name}`);

        // Restore all parameters
        this.params = JSON.parse(JSON.stringify(preset.params)); // Deep clone

        // Map Schrödinger V offset to flowStrength for shader compatibility
        if (this.params.schrodingerVOffset !== undefined) {
            this.params.flowStrength = this.params.schrodingerVOffset;
        }

        // Update all UI controls to reflect loaded values
        this.updateUIFromParams();

        // Update simulations
        if (this.waveSimulation) {
            this.waveSimulation.updateParameters(this.params);

            // If equation type changed, switch to it
            if (this.waveSimulation.currentEquationType !== this.params.equationType) {
                this.waveSimulation.setEquationType(this.params.equationType);
            }
        }

        if (this.renderPipeline) {
            this.renderPipeline.updateParameters(this.params);
        }

        console.log(`Loaded preset: ${preset.name}`);
    }

    deletePreset(index) {
        if (index < 0 || index >= this.presets.length) {
            this.showError('Invalid preset');
            return;
        }

        const preset = this.presets[index];
        console.log(`Deleting preset: ${preset.name}`);
        this.presets.splice(index, 1);
        this.updatePresetList();
    }

    updatePresetList() {
        const presetListContainer = document.getElementById('preset-list');
        if (!presetListContainer) return;

        if (this.presets.length === 0) {
            presetListContainer.innerHTML = '<p class="small-text" style="color: #888; padding: 6px;">No presets saved yet</p>';
            return;
        }

        presetListContainer.innerHTML = this.presets.map((preset, index) => {
            const date = new Date(preset.timestamp);
            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const eqType = preset.params.equationType;
            return `
                <div class="preset-item" style="display: flex; justify-content: space-between; align-items: center; padding: 5px; margin: 3px 0; background: #1a1a1a; border-radius: 3px;">
                    <div style="flex: 1;">
                        <strong style="color: #4fc3f7; font-size: 0.85em;">${preset.name}</strong>
                        <span class="small-text" style="color: #888; margin-left: 6px; font-size: 0.7em;">${timeStr} | ${eqType}</span>
                    </div>
                    <div style="display: flex; gap: 3px;">
                        <button onclick="app.loadPreset(${index})" style="padding: 3px 8px; background: #2d7a4f; border: none; color: white; cursor: pointer; border-radius: 3px; font-size: 0.75em;">Load</button>
                        <button onclick="app.deletePreset(${index})" style="padding: 3px 8px; background: #7a2d2d; border: none; color: white; cursor: pointer; border-radius: 3px; font-size: 0.75em;">Del</button>
                    </div>
                </div>
            `;
        }).join('');
    }

    updateUIFromParams() {
        // Update all sliders and controls to match current params
        const updateSlider = (id, value) => {
            const slider = document.getElementById(id);
            const display = document.getElementById(`${id}-value`);
            if (slider) {
                slider.value = value;
                if (display) {
                    display.textContent = value.toFixed(slider.step < 0.1 ? 3 : (slider.step < 1 ? 2 : 1));
                }
            }
        };

        const updateCheckbox = (id, value) => {
            const checkbox = document.getElementById(id);
            if (checkbox) checkbox.checked = value;
        };

        const updateSelect = (id, value) => {
            const select = document.getElementById(id);
            if (select) select.value = value;
        };

        // Update all controls
        updateSlider('wave-speed', this.params.waveSpeed);
        updateSlider('damping', this.params.damping);
        updateSlider('source-frequency', this.params.sourceFrequency);
        updateSlider('boundary-threshold', this.params.boundaryThreshold);
        updateSlider('wave-amplitude', this.params.waveAmplitude);
        updateSlider('source-strength', this.params.sourceStrength);
        updateSlider('source-size', this.params.sourceSize);
        updateSlider('wave-range', this.params.waveRange);
        updateSlider('steps-per-frame', this.params.stepsPerFrame);
        updateSlider('mix-ratio', this.params.mixRatio);

        updateSlider('cloth-gravity', this.params.clothGravity);
        updateSlider('cloth-stiffness', this.params.clothStiffness);
        updateSlider('cloth-damping', this.params.clothDamping);
        updateSlider('motion-sensitivity', this.params.motionSensitivity);

        updateSlider('fluid-viscosity', this.params.fluidViscosity);
        updateSlider('flow-velocity', this.params.flowVelocity);
        updateSlider('motion-force', this.params.motionForce);
        updateSlider('density-display', this.params.densityDisplay);
        updateSlider('flow-direction', this.params.flowDirection);
        updateSlider('flow-strength', this.params.flowStrength);

        updateSlider('color-scale-min', this.params.colorScaleMin);
        updateSlider('color-scale-max', this.params.colorScaleMax);
        updateSlider('normal-strength', this.params.normalStrength);
        updateSlider('light-direction', this.params.lightDirection);
        updateSlider('light-elevation', this.params.lightElevation);

        updateSlider('aberration-strength', this.params.aberrationStrength);
        updateSlider('aberration-offset', this.params.aberrationOffset);
        updateSlider('aberration-angle', this.params.aberrationAngle);

        updateSlider('schrodinger-prob-scale', this.params.schrodingerProbScale);
        updateSlider('schrodinger-sigma', this.params.schrodingerSigma);
        updateSlider('schrodinger-kx', this.params.schrodingerKx);
        updateSlider('schrodinger-ky', this.params.schrodingerKy);
        updateSlider('schrodinger-v-offset', this.params.schrodingerVOffset);

        updateCheckbox('source-enabled', this.params.sourceEnabled);
        updateCheckbox('normalize-waves', this.params.normalizeWaves);
        updateCheckbox('invert-boundaries', this.params.invertBoundaries);
        updateCheckbox('enable-normal-mapping', this.params.enableNormalMapping);
        updateCheckbox('enable-chromatic-aberration', this.params.enableChromaticAberration);

        updateSelect('equation-type', this.params.equationType);
        updateSelect('color-scheme', this.params.colorScheme);
        updateSelect('blend-mode', this.params.blendMode);
        updateSelect('schrodinger-display-mode', this.params.schrodingerDisplayMode);
    }
}

// The camera is requested only after a click. A presentation slide can
// preload this page, and the browser will not grant the camera until then.
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new WaveCamApp();
    window.app = app;

    const overlay = document.getElementById('start-overlay');
    const start = document.getElementById('start-camera');
    start.addEventListener('click', async () => {
        start.disabled = true;
        start.textContent = 'Ber om kamera…';
        await app.initialize();
        if (app.isRunning) {
            overlay.hidden = true;
        } else {
            start.disabled = false;
            start.textContent = 'Start kamera';
        }
    });
});
