export class ShaderManager {
    constructor() {
        this.customShader = null;
        this.currentEquationType = 'wave';
    }

    getDefaultWaveShader() {
        return `
// Wave equation compute shader
// Implements: ∂²u/∂t² = c²∇²u + f(x,y,t) with Dirichlet boundary conditions

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;  // u(t-1)
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;  // u(t)
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;  // u(t+1)
@group(0) @binding(3) var<storage, read> boundary: array<f32>;  // boundary mask
@group(0) @binding(4) var<uniform> params: SimParams;

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
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    // Boundary check
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Apply Dirichlet boundary conditions
    if (boundary[idx] > params.boundary_threshold) {
        u_next[idx] = 0.0;
        return;
    }
    
    // Calculate Laplacian using finite differences
    let dx2 = params.dx * params.dx;
    let c2 = params.wave_speed * params.wave_speed;
    let dt2 = params.dt * params.dt;
    
    var laplacian = 0.0;
    
    // Central differences for interior points
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        let u_center = u_curr[idx];
        let u_left = u_curr[y * width + (x - 1)];
        let u_right = u_curr[y * width + (x + 1)];
        let u_up = u_curr[(y - 1) * width + x];
        let u_down = u_curr[(y + 1) * width + x];
        
        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u_center) / dx2;
    }
    
    // Source term - oscillating point source at center
    var source = 0.0;
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist2 = (f32(x) - center_x) * (f32(x) - center_x) + 
                   (f32(y) - center_y) * (f32(y) - center_y);
        
        // Properly scaled Gaussian source
        let radius = params.source_size; // User-controllable source width
        let amplitude = params.source_strength; // User-controllable amplitude
        // Proper frequency scaling: 2π * frequency * time
        let phase = 2.0 * 3.14159265359 * params.source_freq * params.time;
        source = amplitude * sin(phase) * exp(-dist2 / (radius * radius));
    }
    
    // Wave equation: u_next = 2*u_curr - u_prev + dt²*(c²*∇²u + f)
    let wave_term = c2 * laplacian + source;
    let next_value = 2.0 * u_curr[idx] - u_prev[idx] + dt2 * wave_term;
    
    // Apply damping
    var damped_value = next_value * (1.0 - params.damping);
    
    // Apply normalization if enabled
    if (params.normalize_waves > 0.5) {
        // Clamp values to the specified range
        damped_value = clamp(damped_value, -params.wave_range, params.wave_range);
    }
    
    u_next[idx] = damped_value;
}`;
    }

    getBathymetryWaveShader() {
        return `
// Bathymetric wave equation - wave speed varies with webcam depth
// Dark areas = shallow (slow waves), Bright areas = deep (fast waves)

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;  // u(t-1)
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;  // u(t)
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;  // u(t+1)
@group(0) @binding(3) var<storage, read> boundary: array<f32>;  // depth map from webcam
@group(0) @binding(4) var<uniform> params: SimParams;

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
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    // Boundary check
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Get local depth from webcam (0 = shallow/dark, 1 = deep/bright)
    let depth = boundary[idx];
    
    // Convert depth to wave speed: shallow water = slow waves, deep water = fast waves
    // Using shallow water wave speed: c = sqrt(g * h) where h is depth
    let min_speed = params.wave_speed * 0.1;  // Very shallow areas
    let max_speed = params.wave_speed * 2.0;  // Very deep areas
    let local_speed = min_speed + depth * (max_speed - min_speed);
    
    // Calculate Laplacian using finite differences
    let dx2 = params.dx * params.dx;
    let c2 = local_speed * local_speed;  // Use local wave speed
    let dt2 = params.dt * params.dt;
    
    var laplacian = 0.0;
    
    // Central differences for interior points
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        let u_center = u_curr[idx];
        let u_left = u_curr[y * width + (x - 1)];
        let u_right = u_curr[y * width + (x + 1)];
        let u_up = u_curr[(y - 1) * width + x];
        let u_down = u_curr[(y + 1) * width + x];
        
        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u_center) / dx2;
    }
    
    // Source term - oscillating point source at center
    var source = 0.0;
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist2 = (f32(x) - center_x) * (f32(x) - center_x) + 
                   (f32(y) - center_y) * (f32(y) - center_y);
        
        // Gaussian source
        let radius = params.source_size;
        let amplitude = params.source_strength;
        let phase = 2.0 * 3.14159265359 * params.source_freq * params.time;
        source = amplitude * sin(phase) * exp(-dist2 / (radius * radius));
    }
    
    // Wave equation with variable speed: u_next = 2*u_curr - u_prev + dt²*(c²(x,y)*∇²u + f)
    let wave_term = c2 * laplacian + source;
    let next_value = 2.0 * u_curr[idx] - u_prev[idx] + dt2 * wave_term;
    
    // Apply damping
    var damped_value = next_value * (1.0 - params.damping);
    
    // Apply normalization if enabled
    if (params.normalize_waves > 0.5) {
        damped_value = clamp(damped_value, -params.wave_range, params.wave_range);
    }
    
    u_next[idx] = damped_value;
}`;
    }

    getDiffusionShader() {
        return `
// Diffusion equation compute shader
// Implements: ∂u/∂t = D∇²u + f(x,y,t)

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;
@group(0) @binding(3) var<storage, read> boundary: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

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
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Apply boundary conditions
    if (boundary[idx] > params.boundary_threshold) {
        u_next[idx] = 0.0;
        return;
    }
    
    // Diffusion coefficient (using wave_speed parameter)
    let D = params.wave_speed;
    let dx2 = params.dx * params.dx;
    
    var laplacian = 0.0;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        let u_center = u_curr[idx];
        let u_left = u_curr[y * width + (x - 1)];
        let u_right = u_curr[y * width + (x + 1)];
        let u_up = u_curr[(y - 1) * width + x];
        let u_down = u_curr[(y + 1) * width + x];
        
        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u_center) / dx2;
    }
    
    // Source term
    var source = 0.0;
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist2 = (f32(x) - center_x) * (f32(x) - center_x) + 
                   (f32(y) - center_y) * (f32(y) - center_y);
        
        source = 5.0 * exp(-0.01 * dist2) * 
                (1.0 + 0.5 * sin(params.source_freq * params.time));
    }
    
    // Diffusion equation: u_next = u_curr + dt*(D*∇²u + f)
    u_next[idx] = u_curr[idx] + params.dt * (D * laplacian + source);
}`;
    }

    getReactionDiffusionShader() {
        return `
// Reaction-Diffusion system (Gray-Scott model)
// Implements: ∂u/∂t = Du*∇²u - uv² + F(1-u)
//            ∂v/∂t = Dv*∇²v + uv² - (F+k)v

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;
@group(0) @binding(3) var<storage, read> boundary: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

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
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Apply boundary conditions
    if (boundary[idx] > params.boundary_threshold) {
        u_next[idx] = 0.0;
        return;
    }
    
    // Gray-Scott parameters
    let Du = 0.16;  // Diffusion rate for u
    let Dv = 0.08;  // Diffusion rate for v
    let F = 0.035;  // Feed rate
    let k = 0.065;  // Kill rate
    
    let dx2 = params.dx * params.dx;
    
    // Get current values (u stored in even indices, v in odd indices conceptually)
    // For simplicity, we'll use u_curr as u and u_prev as v
    let u = u_curr[idx];
    let v = u_prev[idx];
    
    var laplacian_u = 0.0;
    var laplacian_v = 0.0;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Laplacian for u
        let u_center = u_curr[idx];
        let u_left = u_curr[y * width + (x - 1)];
        let u_right = u_curr[y * width + (x + 1)];
        let u_up = u_curr[(y - 1) * width + x];
        let u_down = u_curr[(y + 1) * width + x];
        laplacian_u = (u_left + u_right + u_up + u_down - 4.0 * u_center) / dx2;
        
        // Laplacian for v
        let v_center = u_prev[idx];
        let v_left = u_prev[y * width + (x - 1)];
        let v_right = u_prev[y * width + (x + 1)];
        let v_up = u_prev[(y - 1) * width + x];
        let v_down = u_prev[(y + 1) * width + x];
        laplacian_v = (v_left + v_right + v_up + v_down - 4.0 * v_center) / dx2;
    }
    
    // Reaction terms
    let reaction = u * v * v;
    
    // Update equations
    let du_dt = Du * laplacian_u - reaction + F * (1.0 - u);
    let dv_dt = Dv * laplacian_v + reaction - (F + k) * v;
    
    // Store result (alternating between u and v representation)
    u_next[idx] = u + params.dt * du_dt;
}`;
    }

    getClothDraperyShader() {
        return `
// Cloth Drapery Simulation with Motion Response
// Simulates a 2D cloth hanging from above with gravity and spring forces

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

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;
@group(0) @binding(3) var<storage, read> webcam_prev: array<f32>;
@group(0) @binding(4) var<storage, read> webcam_curr: array<f32>;
@group(0) @binding(5) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = params.width;
    let height = params.height;
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    let dt = params.dt;
    let dx = params.dx;
    
    // Cloth physics parameters from UI
    let gravity = params.cloth_gravity;
    let spring_stiffness = params.cloth_stiffness;
    let cloth_damping = params.cloth_damping;
    let motion_sensitivity = params.motion_sensitivity;
    
    // Get current position and velocity (stored as displacement from rest)
    let curr_pos = u_curr[idx];
    let prev_pos = u_prev[idx];
    let velocity = curr_pos - prev_pos;
    
    // Calculate spring forces from neighbors
    var spring_force = 0.0;
    var neighbor_count = 0.0;
    
    // Check all 4 neighbors for spring connections
    if (x > 0) {
        let left_idx = y * width + (x - 1);
        spring_force += spring_stiffness * (u_curr[left_idx] - curr_pos);
        neighbor_count += 1.0;
    }
    if (x < width - 1) {
        let right_idx = y * width + (x + 1);
        spring_force += spring_stiffness * (u_curr[right_idx] - curr_pos);
        neighbor_count += 1.0;
    }
    if (y > 0) {
        let up_idx = (y - 1) * width + x;
        spring_force += spring_stiffness * (u_curr[up_idx] - curr_pos);
        neighbor_count += 1.0;
    }
    if (y < height - 1) {
        let down_idx = (y + 1) * width + x;
        spring_force += spring_stiffness * (u_curr[down_idx] - curr_pos);
        neighbor_count += 1.0;
    }
    
    // Average spring force
    if (neighbor_count > 0.0) {
        spring_force = spring_force / neighbor_count;
    }
    
    // Apply gravity (stronger towards bottom of screen)
    let gravity_force = gravity * f32(y) / f32(height);
    
    // Motion detection for compression
    let motion_diff = abs(webcam_curr[idx] - webcam_prev[idx]);
    let motion_force = motion_sensitivity * motion_diff * 10.0;
    
    // Cloth hangs from top - fix top row
    var next_pos: f32;
    if (y == 0) {
        // Top row is anchored (hanging points)
        next_pos = 0.0;
    } else {
        // Apply physics: F = ma, where a = (spring + gravity + motion) forces
        let total_force = spring_force + gravity_force + motion_force;
        let acceleration = total_force;
        
        // Verlet integration for cloth simulation
        next_pos = 2.0 * curr_pos - prev_pos + acceleration * dt * dt;
        
        // Apply damping
        next_pos = next_pos * cloth_damping;
        
        // Compression effect from motion
        if (motion_diff > 0.1) {
            next_pos += motion_force * 0.5;
        }
    }
    
    // Optional: Add some wave-like behavior for visual interest
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.3; // Upper third for hanging effect
        let dist_x = f32(x) - center_x;
        let dist_y = f32(y) - center_y;
        let dist2 = dist_x * dist_x + dist_y * dist_y;
        let radius = params.source_size * 10.0;
        
        if (dist2 < radius * radius) {
            let phase = 2.0 * 3.14159265359 * params.source_freq * params.time;
            let wave_source = params.source_strength * 0.1 * sin(phase) * exp(-dist2 / (radius * radius));
            next_pos += wave_source;
        }
    }
    
    // Normalize if enabled
    var final_pos = next_pos;
    if (params.normalize_waves > 0.5) {
        final_pos = clamp(final_pos, -params.wave_range, params.wave_range);
    }
    
    u_next[idx] = final_pos;
}
`;
    }

    getNavierStokesFluidShader() {
        return `
// Navier-Stokes Fluid Simulation (inspired by Pavel Dogreat's WebGL Fluid Simulation)
// Implements proper fluid dynamics with velocity fields and pressure solving

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

@group(0) @binding(0) var<storage, read> velocity_x: array<f32>;      // Velocity field X component
@group(0) @binding(1) var<storage, read> velocity_y: array<f32>;      // Velocity field Y component  
@group(0) @binding(2) var<storage, read_write> density: array<f32>;   // Density/dye field
@group(0) @binding(3) var<storage, read> webcam_data: array<f32>;     // Webcam input for interaction
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Fluid parameters
    let viscosity = params.fluid_viscosity;
    let density_dissipation = 1.0 - params.cloth_damping * 0.01; // Use damping for density dissipation
    let velocity_dissipation = 1.0 - params.damping * 0.1;
    let force_strength = params.motion_force;
    let dye_strength = params.density_display;
    
    // Current values
    let curr_density = density[idx];
    let vx = velocity_x[idx];
    let vy = velocity_y[idx];
    
    // Webcam interaction - detect motion
    let webcam_intensity = webcam_data[idx];
    
    // Simple advection: move density along velocity field
    let dt = params.dt;
    let dx = params.dx;
    
    // Trace back along velocity field
    let back_x = f32(x) - vx * dt / dx;
    let back_y = f32(y) - vy * dt / dx;
    
    // Clamp to grid bounds
    let bx = clamp(back_x, 0.5, f32(width) - 1.5);
    let by = clamp(back_y, 0.5, f32(height) - 1.5);
    
    // Bilinear interpolation
    let x0 = u32(floor(bx));
    let x1 = min(x0 + 1, width - 1);
    let y0 = u32(floor(by));
    let y1 = min(y0 + 1, height - 1);
    
    let fx = bx - f32(x0);
    let fy = by - f32(y0);
    
    // Sample density at four corners
    let d00 = density[y0 * width + x0];
    let d10 = density[y0 * width + x1];
    let d01 = density[y1 * width + x0];
    let d11 = density[y1 * width + x1];
    
    // Interpolate
    let d_top = mix(d00, d10, fx);
    let d_bottom = mix(d01, d11, fx);
    let advected_density = mix(d_top, d_bottom, fy);
    
    // Apply dissipation
    var new_density = advected_density * density_dissipation;
    
    // Use boundary buffer as motion detection (contains squared temporal differences)
    let motion_strength = webcam_intensity; // boundary buffer contains motion data
    
    // Enhanced dye injection from webcam motion - create colorful trails
    if (motion_strength > 0.005) { // Lower threshold for more sensitivity
        // Create rainbow dye based on position and time for visual appeal
        let hue_offset = (f32(x) + f32(y)) * 0.01 + params.time * 0.5;
        let motion_intensity = motion_strength * dye_strength * 25.0;
        
        // Add multiple colored dye sources based on motion intensity
        new_density += motion_intensity * (1.0 + sin(hue_offset));
    }
    
    // Add multiple continuous dye injection points when source enabled
    if (params.source_enabled > 0.5) {
        // Main center source
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist_sq = (f32(x) - center_x) * (f32(x) - center_x) + 
                      (f32(y) - center_y) * (f32(y) - center_y);
        let radius = params.source_size * 8.0;
        
        if (dist_sq < radius * radius) {
            let intensity = exp(-dist_sq / (radius * radius * 0.3));
            let phase = params.time * params.source_freq * 6.28318;
            new_density += params.source_strength * intensity * (2.0 + sin(phase)) * 2.0;
        }
        
        // Additional corner sources for more dynamic flow
        let corner_sources = array<vec2<f32>, 4>(
            vec2<f32>(f32(width) * 0.1, f32(height) * 0.1),
            vec2<f32>(f32(width) * 0.9, f32(height) * 0.1),
            vec2<f32>(f32(width) * 0.1, f32(height) * 0.9),
            vec2<f32>(f32(width) * 0.9, f32(height) * 0.9)
        );
        
        for (var i = 0; i < 4; i++) {
            let corner_dist_sq = (f32(x) - corner_sources[i].x) * (f32(x) - corner_sources[i].x) + 
                                (f32(y) - corner_sources[i].y) * (f32(y) - corner_sources[i].y);
            let corner_radius = params.source_size * 4.0;
            
            if (corner_dist_sq < corner_radius * corner_radius) {
                let corner_intensity = exp(-corner_dist_sq / (corner_radius * corner_radius * 0.5));
                let corner_phase = params.time * params.source_freq * 6.28318 + f32(i) * 1.57;
                new_density += params.source_strength * corner_intensity * (1.0 + sin(corner_phase)) * 0.8;
            }
        }
    }
    
    // Continuous dye injection at edges - create boundary sources
    let edge_distance = min(min(f32(x), f32(width) - f32(x)), min(f32(y), f32(height) - f32(y)));
    if (edge_distance < 8.0) {
        let edge_intensity = (8.0 - edge_distance) / 8.0;
        let edge_phase = params.time * 0.5 + (f32(x) + f32(y)) * 0.1;
        new_density += edge_intensity * dye_strength * (0.5 + 0.3 * sin(edge_phase));
    }
    
    // Add turbulence and dye based on motion gradients
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Motion data is now in webcam_data (boundary buffer) - squared temporal differences
        let motion_left = webcam_data[y * width + (x - 1)];
        let motion_right = webcam_data[y * width + (x + 1)];
        let motion_up = webcam_data[(y - 1) * width + x];
        let motion_down = webcam_data[(y + 1) * width + x];
        
        // Calculate motion gradients for directional dye injection
        let motion_gradient_x = (motion_right - motion_left) * 0.5;
        let motion_gradient_y = (motion_down - motion_up) * 0.5;
        let gradient_magnitude = sqrt(motion_gradient_x * motion_gradient_x + motion_gradient_y * motion_gradient_y);
        
        // Strong dye injection at areas with motion gradients
        new_density += gradient_magnitude * force_strength * 15.0;
        
        // Also add direct motion-based dye from neighboring motion
        let avg_motion = (motion_left + motion_right + motion_up + motion_down) * 0.25;
        if (avg_motion > 0.05) { // Lower threshold for squared differences
            new_density += avg_motion * dye_strength * 5.0;
        }
    }
    
    // Ensure reasonable bounds
    new_density = clamp(new_density, 0.0, 10.0);
    
    density[idx] = new_density;
}`;
    }

    getSimplifiedFluidShader() {
        return `
// Simplified Fluid Simulation with Webcam Motion Detection and Intensity Support
// Enhanced with vorticity confinement for turbulence visualization
// Supports both differential (motion) and simple intensity boundary conditions

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

@group(0) @binding(0) var<storage, read> velocity_field: array<f32>;     // Use u_prev as velocity field
@group(0) @binding(1) var<storage, read> density_curr: array<f32>;       // Use u_curr as current density
@group(0) @binding(2) var<storage, read_write> density_next: array<f32>; // Use u_next as next density
@group(0) @binding(3) var<storage, read> input_data: array<f32>;         // Webcam data (motion or intensity)
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Fluid parameters with improved defaults for turbulence
    let viscosity = params.fluid_viscosity * 0.1; // Reduced for better turbulence
    let dissipation = 1.0 - params.cloth_damping * 0.005; // Slower dissipation
    let force_strength = params.motion_force;
    let dye_strength = params.density_display;
    let vorticity_strength = params.wave_speed * 0.5; // Use wave_speed for vorticity confinement
    
    // Current density and velocity
    let curr_density = density_curr[idx];
    let curr_velocity = velocity_field[idx]; // Use velocity field for advection
    
    // Get input data - supports both motion (differential) and intensity modes
    let input_strength = input_data[idx];
    
    // Detect if input is motion data (small values, squared differences) or intensity (0-1 range)
    // Motion data is typically < 0.5 after squaring, intensity is 0-1
    let is_intensity_mode = input_strength > 0.5 || (input_strength > 0.01 && input_strength < 0.99);
    
    // Simple advection with velocity field
    var new_density = curr_density;
    
    // Base flow advection using flow_direction and flow_strength
    let flow_strength_param = params.flow_strength;
    if (flow_strength_param > 0.001) {
        let flow_angle = params.flow_direction * 3.14159 / 180.0;
        let flow_x = cos(flow_angle) * flow_strength_param;
        let flow_y = sin(flow_angle) * flow_strength_param;
        
        var source_x = f32(x) - flow_x * 15.0;
        var source_y = f32(y) - flow_y * 15.0;
        
        source_x = clamp(source_x, 0.0, f32(width) - 1.0);
        source_y = clamp(source_y, 0.0, f32(height) - 1.0);
        
        let x0 = u32(floor(source_x));
        let x1 = min(x0 + 1, width - 1);
        let y0 = u32(floor(source_y));
        let y1 = min(y0 + 1, height - 1);
        
        let fx = source_x - f32(x0);
        let fy = source_y - f32(y0);
        
        let d00 = density_curr[y0 * width + x0];
        let d10 = density_curr[y0 * width + x1];
        let d01 = density_curr[y1 * width + x0];
        let d11 = density_curr[y1 * width + x1];
        
        let d_top = mix(d00, d10, fx);
        let d_bottom = mix(d01, d11, fx);
        let advected_density = mix(d_top, d_bottom, fy);
        
        new_density = mix(new_density, advected_density, flow_strength_param * 2.5);
    }
    
    // Vorticity confinement for turbulence enhancement
    var vorticity_force_x = 0.0;
    var vorticity_force_y = 0.0;
    
    if (x > 1 && x < width - 2 && y > 1 && y < height - 2) {
        // Calculate curl (vorticity) from density gradients
        let d_left = density_curr[y * width + (x - 1)];
        let d_right = density_curr[y * width + (x + 1)];
        let d_up = density_curr[(y - 1) * width + x];
        let d_down = density_curr[(y + 1) * width + x];
        
        // Curl approximation from density field
        let curl = (d_down - d_up) - (d_right - d_left);
        
        // Calculate curl gradient for vorticity confinement
        let curl_left = (density_curr[(y + 1) * width + (x - 1)] - density_curr[(y - 1) * width + (x - 1)]) - 
                        (density_curr[y * width + x] - density_curr[y * width + (x - 2)]);
        let curl_right = (density_curr[(y + 1) * width + (x + 1)] - density_curr[(y - 1) * width + (x + 1)]) - 
                         (density_curr[y * width + (x + 2)] - density_curr[y * width + x]);
        let curl_up = (density_curr[(y - 1) * width + (x + 1)] - density_curr[(y - 1) * width + (x - 1)]) - 
                      (density_curr[y * width + x] - density_curr[(y - 2) * width + x]);
        let curl_down = (density_curr[(y + 1) * width + (x + 1)] - density_curr[(y + 1) * width + (x - 1)]) - 
                        (density_curr[(y + 2) * width + x] - density_curr[y * width + x]);
        
        // Gradient of curl magnitude
        let curl_grad_x = (abs(curl_right) - abs(curl_left)) * 0.5;
        let curl_grad_y = (abs(curl_down) - abs(curl_up)) * 0.5;
        let curl_grad_mag = sqrt(curl_grad_x * curl_grad_x + curl_grad_y * curl_grad_y);
        
        // Vorticity confinement force
        if (curl_grad_mag > 0.001) {
            let norm_x = curl_grad_x / curl_grad_mag;
            let norm_y = curl_grad_y / curl_grad_mag;
            
            vorticity_force_x = norm_y * curl * vorticity_strength;
            vorticity_force_y = -norm_x * curl * vorticity_strength;
        }
        
        // Apply vorticity-based advection
        if (abs(vorticity_force_x) > 0.01 || abs(vorticity_force_y) > 0.01) {
            let vort_source_x = clamp(f32(x) - vorticity_force_x * 2.0, 0.0, f32(width) - 1.0);
            let vort_source_y = clamp(f32(y) - vorticity_force_y * 2.0, 0.0, f32(height) - 1.0);
            
            let vx0 = u32(floor(vort_source_x));
            let vx1 = min(vx0 + 1, width - 1);
            let vy0 = u32(floor(vort_source_y));
            let vy1 = min(vy0 + 1, height - 1);
            
            let vfx = vort_source_x - f32(vx0);
            let vfy = vort_source_y - f32(vy0);
            
            let vd00 = density_curr[vy0 * width + vx0];
            let vd10 = density_curr[vy0 * width + vx1];
            let vd01 = density_curr[vy1 * width + vx0];
            let vd11 = density_curr[vy1 * width + vx1];
            
            let vd_top = mix(vd00, vd10, vfx);
            let vd_bottom = mix(vd01, vd11, vfx);
            let vorticity_density = mix(vd_top, vd_bottom, vfy);
            
            new_density = mix(new_density, vorticity_density, 0.3);
        }
        
        // Diffusion with neighbors (reduced for turbulence)
        let left = density_curr[y * width + (x - 1)];
        let right = density_curr[y * width + (x + 1)];
        let up = density_curr[(y - 1) * width + x];
        let down = density_curr[(y + 1) * width + x];
        let avg = (left + right + up + down) * 0.25;
        
        new_density = mix(new_density, avg, viscosity);
    }
    
    // Apply dissipation
    new_density *= dissipation;
    
    // Handle both intensity and motion input modes
    if (is_intensity_mode) {
        // Intensity mode: use webcam intensity directly as dye source
        if (input_strength > params.boundary_threshold) {
            let intensity_contribution = (input_strength - params.boundary_threshold) / (1.0 - params.boundary_threshold);
            new_density += intensity_contribution * dye_strength * 3.0;
            
            // Add swirling motion from intensity gradients
            let int_left = input_data[y * width + (x - 1)];
            let int_right = input_data[y * width + (x + 1)];
            let int_up = input_data[(y - 1) * width + x];
            let int_down = input_data[(y + 1) * width + x];
            
            let int_grad_x = (int_right - int_left) * 0.5;
            let int_grad_y = (int_down - int_up) * 0.5;
            
            // Add perpendicular forces for swirl
            if (abs(int_grad_x) > 0.01 || abs(int_grad_y) > 0.01) {
                new_density += abs(int_grad_x + int_grad_y) * force_strength * 2.0;
            }
        }
    } else {
        // Motion mode: use temporal differences
        if (input_strength > 0.0001) {
            // Direct dye injection from motion
            new_density += input_strength * dye_strength * force_strength * 5.0;
            
            // Motion-based turbulence
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                let motion_left = input_data[y * width + (x - 1)];
                let motion_right = input_data[y * width + (x + 1)];
                let motion_up = input_data[(y - 1) * width + x];
                let motion_down = input_data[(y + 1) * width + x];
                
                let motion_grad_x = (motion_right - motion_left) * 0.5;
                let motion_grad_y = (motion_down - motion_up) * 0.5;
                let grad_mag = sqrt(motion_grad_x * motion_grad_x + motion_grad_y * motion_grad_y);
                
                // Enhanced turbulent dye injection at motion gradients
                new_density += grad_mag * force_strength * 10.0;
                
                // Create swirling patterns from motion
                if (grad_mag > 0.01) {
                    let swirl_x = f32(x) - motion_grad_y * force_strength * 8.0;
                    let swirl_y = f32(y) + motion_grad_x * force_strength * 8.0;
                    
                    if (swirl_x >= 0.0 && swirl_x < f32(width) - 1.0 && 
                        swirl_y >= 0.0 && swirl_y < f32(height) - 1.0) {
                        let sx = u32(swirl_x);
                        let sy = u32(swirl_y);
                        let swirl_density = density_curr[sy * width + sx];
                        new_density = mix(new_density, swirl_density, input_strength * 1.5);
                    }
                }
            }
        }
    }
    
    // Add continuous dye source at center
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist_sq = (f32(x) - center_x) * (f32(x) - center_x) + 
                      (f32(y) - center_y) * (f32(y) - center_y);
        let radius = params.source_size * 8.0;
        
        if (dist_sq < radius * radius) {
            let intensity = exp(-dist_sq / (radius * radius * 0.4));
            let phase = params.time * params.source_freq * 6.28318;
            new_density += params.source_strength * intensity * (1.5 + sin(phase)) * 2.0;
        }
    }
    
    // Clamp to reasonable range
    new_density = clamp(new_density, 0.0, 12.0);
    
    density_next[idx] = new_density;
}`;
    }

    getVelocityAdvectionShader() {
        return `
// Velocity Advection Shader - advects velocity field and adds forces from webcam motion

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

@group(0) @binding(0) var<storage, read> velocity_x_in: array<f32>;
@group(0) @binding(1) var<storage, read> velocity_y_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_x_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_y_out: array<f32>;
@group(0) @binding(4) var<storage, read> webcam_prev: array<f32>;
@group(0) @binding(5) var<storage, read> webcam_curr: array<f32>;
@group(0) @binding(6) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Fluid parameters
    let velocity_dissipation = 1.0 - params.damping * 0.01;
    let force_strength = params.motion_force;
    let dt = params.dt;
    let dx = params.dx;
    
    // Current velocity
    let vx = velocity_x_in[idx];
    let vy = velocity_y_in[idx];
    
    // Self-advection: trace back along velocity field
    let back_x = f32(x) - vx * dt / dx;
    let back_y = f32(y) - vy * dt / dx;
    
    // Clamp to grid bounds
    let bx = clamp(back_x, 0.5, f32(width) - 1.5);
    let by = clamp(back_y, 0.5, f32(height) - 1.5);
    
    // Bilinear interpolation for velocity
    let x0 = u32(floor(bx));
    let x1 = min(x0 + 1, width - 1);
    let y0 = u32(floor(by));
    let y1 = min(y0 + 1, height - 1);
    
    let fx = bx - f32(x0);
    let fy = by - f32(y0);
    
    // Sample velocity at four corners
    let vx00 = velocity_x_in[y0 * width + x0];
    let vx10 = velocity_x_in[y0 * width + x1];
    let vx01 = velocity_x_in[y1 * width + x0];
    let vx11 = velocity_x_in[y1 * width + x1];
    
    let vy00 = velocity_y_in[y0 * width + x0];
    let vy10 = velocity_y_in[y0 * width + x1];
    let vy01 = velocity_y_in[y1 * width + x0];
    let vy11 = velocity_y_in[y1 * width + x1];
    
    // Interpolate velocity
    let vx_top = mix(vx00, vx10, fx);
    let vx_bottom = mix(vx01, vx11, fx);
    var new_vx = mix(vx_top, vx_bottom, fy);
    
    let vy_top = mix(vy00, vy10, fx);
    let vy_bottom = mix(vy01, vy11, fx);
    var new_vy = mix(vy_top, vy_bottom, fy);
    
    // Apply dissipation
    new_vx *= velocity_dissipation;
    new_vy *= velocity_dissipation;
    
    // Add forces from webcam motion (temporal difference - frame-to-frame changes)
    let motion_diff = webcam_curr[idx] - webcam_prev[idx];
    let motion_magnitude = abs(motion_diff);
    
    // Add velocity based on MOTION gradients and curl computation for turbulence
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Calculate motion gradients (temporal differences in neighboring pixels)
        let motion_left = abs(webcam_curr[y * width + (x - 1)] - webcam_prev[y * width + (x - 1)]);
        let motion_right = abs(webcam_curr[y * width + (x + 1)] - webcam_prev[y * width + (x + 1)]);
        let motion_up = abs(webcam_curr[(y - 1) * width + x] - webcam_prev[(y - 1) * width + x]);
        let motion_down = abs(webcam_curr[(y + 1) * width + x] - webcam_prev[(y + 1) * width + x]);
        
        // Motion gradient - direction of strongest motion
        let motion_grad_x = (motion_right - motion_left) * 0.5;
        let motion_grad_y = (motion_down - motion_up) * 0.5;
        
        // Calculate velocity curl (vorticity) for turbulence enhancement
        let vx_left = velocity_x_in[y * width + (x - 1)];
        let vx_right = velocity_x_in[y * width + (x + 1)];
        let vy_up = velocity_y_in[(y - 1) * width + x];
        let vy_down = velocity_y_in[(y + 1) * width + x];
        
        // Curl = ∂vy/∂x - ∂vx/∂y (2D vorticity)
        let curl = ((vy_down - vy_up) * 0.5) - ((vx_right - vx_left) * 0.5);
        let curl_magnitude = abs(curl);
        
        // Add force proportional to motion gradients
        new_vx += motion_grad_x * force_strength * 50.0;
        new_vy += motion_grad_y * force_strength * 50.0;
        
        // Add direct motion force - push fluid away from moving areas
        if (motion_magnitude > 0.05) {
            new_vx += motion_grad_x * motion_magnitude * force_strength * 100.0;
            new_vy += motion_grad_y * motion_magnitude * force_strength * 100.0;
            
            // Enhanced swirl/turbulence based on both motion and existing curl
            let base_swirl = motion_magnitude * force_strength * 20.0;
            let curl_enhancement = curl_magnitude * force_strength * 15.0;
            let total_swirl = base_swirl + curl_enhancement;
            
            // Add perpendicular forces for vortex formation
            new_vx += motion_grad_y * total_swirl;
            new_vy -= motion_grad_x * total_swirl;
            
            // Add curl-based forces to maintain and enhance vortices
            if (curl_magnitude > 0.1) {
                let curl_sign = sign(curl);
                let vortex_strength = curl_magnitude * force_strength * 25.0;
                
                // Add forces perpendicular to position vector from vortex center
                let center_offset_x = f32(x) - f32(width) * 0.5;
                let center_offset_y = f32(y) - f32(height) * 0.5;
                let offset_magnitude = sqrt(center_offset_x * center_offset_x + center_offset_y * center_offset_y);
                
                if (offset_magnitude > 1.0) {
                    new_vx += (-center_offset_y / offset_magnitude) * curl_sign * vortex_strength;
                    new_vy += (center_offset_x / offset_magnitude) * curl_sign * vortex_strength;
                }
            }
        }
    }
    
    // Add directional base flow based on flow_direction and flow_strength
    let flow_angle = params.flow_direction * 3.14159 / 180.0; // Convert degrees to radians
    let flow_x = cos(flow_angle) * params.flow_strength * 20.0;
    let flow_y = sin(flow_angle) * params.flow_strength * 20.0;
    new_vx += flow_x;
    new_vy += flow_y;
    
    // Add multiple oscillating forces for more interesting fluid dynamics
    if (params.source_enabled > 0.5) {
        // Main center source - radial oscillation
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dx_center = f32(x) - center_x;
        let dy_center = f32(y) - center_y;
        let dist_sq = dx_center * dx_center + dy_center * dy_center;
        let radius = params.source_size * 10.0;
        
        if (dist_sq < radius * radius && dist_sq > 1.0) {
            let dist = sqrt(dist_sq);
            let intensity = exp(-dist_sq / (radius * radius * 0.5));
            let phase = params.time * params.source_freq * 6.28318;
            let force_magnitude = sin(phase) * params.source_strength * intensity * 60.0;
            
            // Radial force with rotational component
            new_vx += (dx_center / dist) * force_magnitude;
            new_vy += (dy_center / dist) * force_magnitude;
            
            // Add rotational component for vortex-like motion
            let rotation_force = cos(phase + 1.57) * params.source_strength * intensity * 30.0;
            new_vx += (-dy_center / dist) * rotation_force;
            new_vy += (dx_center / dist) * rotation_force;
        }
        
        // Additional pulsing sources at quarter positions
        let quarter_sources = array<vec2<f32>, 4>(
            vec2<f32>(f32(width) * 0.25, f32(height) * 0.25),
            vec2<f32>(f32(width) * 0.75, f32(height) * 0.25),
            vec2<f32>(f32(width) * 0.25, f32(height) * 0.75),
            vec2<f32>(f32(width) * 0.75, f32(height) * 0.75)
        );
        
        for (var i = 0; i < 4; i++) {
            let dx_quarter = f32(x) - quarter_sources[i].x;
            let dy_quarter = f32(y) - quarter_sources[i].y;
            let quarter_dist_sq = dx_quarter * dx_quarter + dy_quarter * dy_quarter;
            let quarter_radius = params.source_size * 6.0;
            
            if (quarter_dist_sq < quarter_radius * quarter_radius && quarter_dist_sq > 1.0) {
                let quarter_dist = sqrt(quarter_dist_sq);
                let quarter_intensity = exp(-quarter_dist_sq / (quarter_radius * quarter_radius * 0.3));
                let quarter_phase = params.time * params.source_freq * 6.28318 + f32(i) * 1.57;
                let quarter_force = sin(quarter_phase) * params.source_strength * quarter_intensity * 30.0;
                
                // Alternating radial and tangential forces for complex flow patterns
                if (i % 2 == 0) {
                    new_vx += (dx_quarter / quarter_dist) * quarter_force;
                    new_vy += (dy_quarter / quarter_dist) * quarter_force;
                } else {
                    new_vx += (-dy_quarter / quarter_dist) * quarter_force;
                    new_vy += (dx_quarter / quarter_dist) * quarter_force;
                }
            }
        }
    }
    
    // Improved boundary conditions - mix of no-slip and free-slip for more natural flow
    let boundary_damping = 0.8; // Partial damping instead of complete stopping
    let boundary_thickness = 3.0; // Gradual boundary transition
    
    // Left and right boundaries
    if (f32(x) < boundary_thickness) {
        let boundary_factor = f32(x) / boundary_thickness;
        new_vx *= boundary_factor * boundary_damping;
        // Allow some tangential flow along boundaries
        new_vy *= (0.5 + 0.5 * boundary_factor);
    } else if (f32(x) > f32(width) - boundary_thickness - 1.0) {
        let boundary_factor = (f32(width) - f32(x) - 1.0) / boundary_thickness;
        new_vx *= boundary_factor * boundary_damping;
        new_vy *= (0.5 + 0.5 * boundary_factor);
    }
    
    // Top and bottom boundaries  
    if (f32(y) < boundary_thickness) {
        let boundary_factor = f32(y) / boundary_thickness;
        new_vy *= boundary_factor * boundary_damping;
        // Allow some tangential flow along boundaries
        new_vx *= (0.5 + 0.5 * boundary_factor);
    } else if (f32(y) > f32(height) - boundary_thickness - 1.0) {
        let boundary_factor = (f32(height) - f32(y) - 1.0) / boundary_thickness;
        new_vy *= boundary_factor * boundary_damping;
        new_vx *= (0.5 + 0.5 * boundary_factor);
    }
    
    // Clamp velocities to reasonable range
    new_vx = clamp(new_vx, -100.0, 100.0);
    new_vy = clamp(new_vy, -100.0, 100.0);
    
    velocity_x_out[idx] = new_vx;
    velocity_y_out[idx] = new_vy;
}`;
    }

    getLatticeBoltzmannShader() {
        return `
// Simplified Lattice Boltzmann Method (LBM) 
// Basic fluid simulation without motion detection (for now)

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

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;
@group(0) @binding(3) var<storage, read> boundary: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

// D2Q9 Lattice Boltzmann weights and velocities
const w = array<f32, 9>(
    4.0/9.0,  // center
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,  // cardinal directions
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0  // diagonal directions
);

// Velocity vectors for D2Q9
const e_x = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const e_y = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);

// Opposite directions for bounce-back boundary conditions
const opp = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

fn equilibrium(rho: f32, ux: f32, uy: f32, i: u32) -> f32 {
    let ex = f32(e_x[i]);
    let ey = f32(e_y[i]);
    let eu = ex * ux + ey * uy;
    let u2 = ux * ux + uy * uy;
    
    return rho * w[i] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = params.width;
    let height = params.height;
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // LBM parameters - map UI controls to fluid physics
    let viscosity = params.fluid_viscosity; // Fluid Viscosity slider
    let base_flow = params.flow_velocity; // Flow Velocity slider  
    let motion_force = params.motion_force; // Motion Force slider
    let density_scale = params.density_display; // Density Display slider
    
    // Simplified LBM: use wave buffers for density evolution
    let rho_old = u_prev[idx];
    let rho_curr = u_curr[idx];
    
    // Get webcam-based boundary and use it for fluid interaction
    let boundary_strength = boundary[idx];
    let is_boundary = boundary_strength > params.boundary_threshold;
    
    // Initialize new density value
    var rho_new = rho_old;
    
    // Webcam interaction: use boundary strength as fluid disturbance
    let webcam_influence = boundary_strength * motion_force;
    
    // Boundary conditions - walls block flow but create turbulence
    if (is_boundary || x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        // Boundaries create density disturbances
        rho_new = 0.5 + webcam_influence * 0.5;
    } else {
        // Fluid simulation in free space
        var neighbor_avg = 0.0;
        var neighbor_count = 0.0;
        
        // Sample 4-connected neighbors with webcam influence
        if (x > 0) {
            let left_boundary = boundary[y * width + (x - 1)];
            let left_rho = u_prev[y * width + (x - 1)];
            // Much stronger influence from areas with webcam activity
            let influence = 1.0 + left_boundary * motion_force * 0.5;
            neighbor_avg += left_rho * influence;
            neighbor_count += influence;
        }
        if (x < width - 1) {
            let right_boundary = boundary[y * width + (x + 1)];
            let right_rho = u_prev[y * width + (x + 1)];
            let influence = 1.0 + right_boundary * motion_force * 0.5;
            neighbor_avg += right_rho * influence;
            neighbor_count += influence;
        }
        if (y > 0) {
            let up_boundary = boundary[(y - 1) * width + x];
            let up_rho = u_prev[(y - 1) * width + x];
            let influence = 1.0 + up_boundary * motion_force * 0.5;
            neighbor_avg += up_rho * influence;
            neighbor_count += influence;
        }
        if (y < height - 1) {
            let down_boundary = boundary[(y + 1) * width + x];
            let down_rho = u_prev[(y + 1) * width + x];
            let influence = 1.0 + down_boundary * motion_force * 0.5;
            neighbor_avg += down_rho * influence;
            neighbor_count += influence;
        }
        
        if (neighbor_count > 0.0) {
            neighbor_avg /= neighbor_count;
            // More aggressive viscosity-controlled diffusion
            rho_new = mix(rho_old, neighbor_avg, viscosity * 2.0);
        }
        
        // Stronger base flow (left to right)
        if (x > 1 && base_flow > 0.001) {
            let flow_source = u_prev[y * width + (x - 1)];
            rho_new = mix(rho_new, flow_source, base_flow * 0.5);
        }
        
        // Much stronger webcam influence - creates visible density sources
        rho_new += webcam_influence * 2.0;
        
        // Add much stronger oscillating source at center
        if (params.source_enabled > 0.5) {
            let center_x = f32(width) * 0.5;
            let center_y = f32(height) * 0.5;
            let dx = f32(x) - center_x;
            let dy = f32(y) - center_y;
            let dist2 = dx * dx + dy * dy;
            let radius = params.source_size * 10.0;  // Larger radius
            
            if (dist2 < radius * radius) {
                let phase = 2.0 * 3.14159 * params.source_freq * params.time;
                // Much stronger source
                let source_strength = params.source_strength * 2.0 * sin(phase) * exp(-dist2 / (radius * radius));
                rho_new += source_strength;
            }
        }
    }
    
    // Ensure density stays positive and bounded
    rho_new = clamp(rho_new, 0.0, 3.0);
    
    // Store results
    // Use density scale parameter for visualization
    var output_value = rho_new * density_scale;
    
    // Debug: Show test pattern to verify shader is working
    if (params.source_enabled > 0.5) {
        // Add a much more visible moving test pattern
        let test_x = f32(x) / f32(width);
        let test_y = f32(y) / f32(height);
        let test_pattern = sin(test_x * 8.0 + params.time * 2.0) * sin(test_y * 8.0 + params.time * 2.0) * 2.0 + 2.0;
        output_value += test_pattern;
        
        // Also add a more visible grid to verify coordinates
        let grid_x = step(0.9, fract(f32(x) / 20.0));
        let grid_y = step(0.9, fract(f32(y) / 20.0));
        output_value += (grid_x + grid_y) * 0.5;
    }
    
    // Make sure we have much more visible output for LBM
    output_value = max(output_value, 1.0);  // Higher baseline
    
    // Scale up the output significantly for better visibility
    output_value *= 3.0;
    
    u_next[idx] = output_value;
    
    // Note: In a full LBM implementation, we would store all 9 distribution functions
    // and perform proper streaming and collision steps. This is a simplified version
    // that captures the essential fluid-like behavior with motion response.
}
`;
    }

    getShaderForEquationType(equationType) {
        switch (equationType) {
            case 'wave':
                return this.customShader || this.getDefaultWaveShader();
            case 'wave-bathymetry':
                return this.getBathymetryWaveShader();
            case 'wave-motion':
                return this.getMotionWaveShader();
            case 'cloth-drapery':
                return this.getClothDraperyShader();
            case 'lattice-boltzmann':
                return this.getLatticeBoltzmannShader();
            case 'navier-stokes-fluid':
                return this.getSimplifiedFluidShader();
            case 'velocity-advection':
                return this.getVelocityAdvectionShader();
            case 'schrodinger':
                return this.getSchrodingerShader();
            case 'source-debug':
                return this.getSourceDebugShader();
            case 'diffusion':
                return this.getDiffusionShader();
            case 'reaction-diffusion':
                return this.getReactionDiffusionShader();
            default:
                return this.getDefaultWaveShader();
        }
    }

    setCustomShader(shaderCode) {
        // Basic validation
        if (!shaderCode.includes('@compute')) {
            throw new Error('Shader must include @compute entry point');
        }

        if (!shaderCode.includes('main(')) {
            throw new Error('Shader must include main() function');
        }

        this.customShader = shaderCode;
    }

    clearCustomShader() {
        this.customShader = null;
    }

    hasCustomShader() {
        return this.customShader !== null;
    }

    getMotionWaveShader() {
        return `
// Simple Motion-based Wave Equation - Based on Hard Boundary Wave Shader
// Uses webcam temporal difference as direct wave source
// Clean implementation following the proven hard boundary pattern

@group(0) @binding(0) var<storage, read> u_prev: array<f32>;     // u(t-1)
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;     // u(t)
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>; // u(t+1)
@group(0) @binding(3) var<storage, read> webcam_prev: array<f32>; // previous webcam frame
@group(0) @binding(4) var<storage, read> webcam_curr: array<f32>; // current webcam frame
@group(0) @binding(5) var<uniform> params: SimParams;

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
    let width = params.width;
    let height = params.height;
    
    // Boundary check
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Calculate Laplacian using finite differences (same as hard boundary shader)
    let dx2 = params.dx * params.dx;
    let c2 = params.wave_speed * params.wave_speed;
    let dt2 = params.dt * params.dt;
    
    var laplacian = 0.0;
    
    // Central differences for interior points
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        let u_center = u_curr[idx];
        let u_left = u_curr[y * width + (x - 1)];
        let u_right = u_curr[y * width + (x + 1)];
        let u_up = u_curr[(y - 1) * width + x];
        let u_down = u_curr[(y + 1) * width + x];
        
        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u_center) / dx2;
    }
    
    // Motion-based source term - SIMPLE AND DIRECT
    var source = 0.0;
    
    // Calculate temporal difference (motion detection)
    let motion_diff = webcam_curr[idx] - webcam_prev[idx];
    let motion_magnitude = abs(motion_diff);
    
    // Use motion as direct source (like the hard boundary shader uses its source)
    // Simple, clean motion detection - no debug oscillations
    source = motion_magnitude * params.motion_force * params.source_strength * 100.0;
    
    // Optional: Add traditional oscillating source at center (like hard boundary shader)
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist2 = (f32(x) - center_x) * (f32(x) - center_x) + 
                   (f32(y) - center_y) * (f32(y) - center_y);
        
        // Properly scaled Gaussian source (same as hard boundary)
        let radius = params.source_size;
        let amplitude = params.source_strength;
        let phase = 2.0 * 3.14159265359 * params.source_freq * params.time;
        let center_source = amplitude * sin(phase) * exp(-dist2 / (radius * radius));
        
        source += center_source;
    }
    
    // Wave equation: u_next = 2*u_curr - u_prev + dt²*(c²*∇²u + f) (same as hard boundary)
    let wave_term = c2 * laplacian + source;
    let next_value = 2.0 * u_curr[idx] - u_prev[idx] + dt2 * wave_term;
    
    // Apply damping (same as hard boundary)
    var damped_value = next_value * (1.0 - params.damping);
    
    // Apply normalization if enabled (same as hard boundary)
    if (params.normalize_waves > 0.5) {
        damped_value = clamp(damped_value, -params.wave_range, params.wave_range);
    }
    
    u_next[idx] = damped_value;
}`;
    }

    getSchrodingerShader() {
        return `
// 2D Time-Dependent Schrödinger Equation using FDTD (Finite Difference Time Domain)
// Implements: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + V(x,y)ψ
// Split into real and imaginary parts with leapfrog time-stepping for stability
//
// Potential V(x,y) from webcam:
//   V_raw = (1 - intensity)² × boundary_threshold    [potential extraction]
//   V(x,y) = V_amplitude × (V_raw - V_offset)        [Schrödinger shader]
//
// Where:
//   - intensity = grayscale webcam (0=black, 1=white)
//   - boundary_threshold = base scaling (UI: "Boundary Threshold")
//   - V_amplitude = cloth_gravity (UI: "Potential Amplitude")
//   - V_offset = flow_strength (UI: "Potential Offset")
//   - Dark areas create high potential barriers (by default)
//   - Negative offset creates attractive potential wells

struct SimParams {
    width: u32,
    height: u32,
    dt: f32,
    dx: f32,
    wave_speed: f32,        // Used as ℏ/m coefficient
    damping: f32,           // Wavefunction damping (for visualization stability)
    source_freq: f32,       // Initial wavepacket momentum
    source_enabled: f32,    // Enable/disable wavepacket source
    time: f32,
    boundary_threshold: f32, // Base potential normalization from webcam
    source_strength: f32,    // Wavepacket amplitude
    source_size: f32,        // Wavepacket width (sigma)
    normalize_waves: f32,
    wave_range: f32,
    cloth_gravity: f32,      // Schrödinger: Potential amplitude (V_amplitude)
    cloth_stiffness: f32,    // Schrödinger: Wavepacket initial momentum kx
    cloth_damping: f32,      // Schrödinger: Wavepacket initial momentum ky
    motion_sensitivity: f32, // Schrödinger: Visualization gamma for contrast
    invert_boundaries: f32,  // Invert dark/bright potential mapping
    fluid_viscosity: f32,
    flow_velocity: f32,
    motion_force: f32,
    density_display: f32,
    flow_direction: f32,
    flow_strength: f32       // Schrödinger: Potential offset (V_offset)
}

// Schrödinger equation uses 6-binding layout:
// binding 0: ψ_R (real part) at time t - read only
// binding 1: ψ_I (imaginary part) at time t - read only
// binding 2: output buffer for visualization - write
// binding 3: ψ_R_next - stores updated real part
// binding 4: ψ_I_next - stores updated imaginary part  
// binding 5: V(x,y) potential from webcam - read only
// binding 6: params - uniform

@group(0) @binding(0) var<storage, read> psi_R: array<f32>;      // Real part ψ_R(t)
@group(0) @binding(1) var<storage, read> psi_I: array<f32>;      // Imaginary part ψ_I(t)
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Visualization output
@group(0) @binding(3) var<storage, read_write> psi_R_next: array<f32>; // Real part ψ_R(t+dt)
@group(0) @binding(4) var<storage, read_write> psi_I_next: array<f32>; // Imaginary part ψ_I(t+dt)
@group(0) @binding(5) var<storage, read> potential: array<f32>;  // V(x,y) from webcam
@group(0) @binding(6) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    // Boundary check
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Physical parameters (normalized units: ℏ = 1, m = 1/2)
    // So the kinetic term coefficient is ℏ²/(2m) = 1/(2*0.5) = 1
    let hbar_over_m = params.wave_speed;  // Use wave_speed as ℏ/m tuning
    let dt = params.dt * 0.5;  // Smaller timestep for Schrödinger stability
    let dx2 = params.dx * params.dx;
    
    // Get potential from webcam with amplitude*(V - offset) formula
    let V_raw = potential[idx];
    let V_amplitude = params.cloth_gravity;    // Potential amplitude from UI
    let V_offset = params.flow_strength;        // Potential offset from UI (repurposed)
    let V = V_amplitude * (V_raw - V_offset);
    
    // Current wavefunction values
    let R = psi_R[idx];
    let I = psi_I[idx];
    
    // Calculate Laplacian using 5-point stencil for both components
    var laplacian_R = 0.0;
    var laplacian_I = 0.0;
    
    // Absorbing boundary conditions at edges
    let boundary_width = 10.0;
    let dist_left = f32(x);
    let dist_right = f32(width - 1 - x);
    let dist_top = f32(y);
    let dist_bottom = f32(height - 1 - y);
    let min_dist = min(min(dist_left, dist_right), min(dist_top, dist_bottom));
    
    // Absorbing factor (1 in interior, smoothly goes to 0 at boundaries)
    var absorb = 1.0;
    if (min_dist < boundary_width) {
        absorb = min_dist / boundary_width;
        absorb = absorb * absorb;  // Quadratic falloff
    }
    
    // Interior points: compute Laplacian
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Laplacian of ψ_R
        let R_center = R;
        let R_left = psi_R[y * width + (x - 1)];
        let R_right = psi_R[y * width + (x + 1)];
        let R_up = psi_R[(y - 1) * width + x];
        let R_down = psi_R[(y + 1) * width + x];
        laplacian_R = (R_left + R_right + R_up + R_down - 4.0 * R_center) / dx2;
        
        // Laplacian of ψ_I
        let I_center = I;
        let I_left = psi_I[y * width + (x - 1)];
        let I_right = psi_I[y * width + (x + 1)];
        let I_up = psi_I[(y - 1) * width + x];
        let I_down = psi_I[(y + 1) * width + x];
        laplacian_I = (I_left + I_right + I_up + I_down - 4.0 * I_center) / dx2;
    }
    
    // Schrödinger equation split into real/imaginary:
    // ∂ψ_R/∂t = (ℏ/2m)∇²ψ_I - (V/ℏ)ψ_I
    // ∂ψ_I/∂t = -(ℏ/2m)∇²ψ_R + (V/ℏ)ψ_R
    // In our normalized units (ℏ=1):
    // ∂ψ_R/∂t = (hbar_over_m/2)∇²ψ_I - V*ψ_I
    // ∂ψ_I/∂t = -(hbar_over_m/2)∇²ψ_R + V*ψ_R
    
    let kinetic_coeff = hbar_over_m * 0.5;
    
    // Time derivatives
    let dR_dt = kinetic_coeff * laplacian_I - V * I;
    let dI_dt = -kinetic_coeff * laplacian_R + V * R;
    
    // Explicit Euler update
    var R_new = R + dt * dR_dt;
    var I_new = I + dt * dI_dt;
    
    // Apply absorbing boundaries
    R_new *= absorb;
    I_new *= absorb;
    
    // Apply global damping only if damping parameter is set
    if (params.damping > 0.001) {
        let damping_factor = 1.0 - params.damping * 0.1;
        R_new *= damping_factor;
        I_new *= damping_factor;
    }
    
    // Gaussian wavepacket source
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dx_src = f32(x) - center_x;
        let dy_src = f32(y) - center_y;
        let dist2 = dx_src * dx_src + dy_src * dy_src;
        
        // Wavepacket parameters
        let sigma = params.source_size * 2.0;  // Width of Gaussian envelope
        let sigma2 = sigma * sigma;
        let amplitude = params.source_strength * 0.5;
        
        // Initial momentum (k-vector)
        let kx = params.cloth_stiffness * 0.5;  // Momentum in x
        let ky = params.cloth_damping * 0.5;    // Momentum in y
        
        // Oscillating Gaussian wavepacket: A * exp(-r²/2σ²) * exp(i(k·r - ωt))
        // Real part: A * exp(-r²/2σ²) * cos(k·r - ωt)
        // Imag part: A * exp(-r²/2σ²) * sin(k·r - ωt)
        let envelope = amplitude * exp(-dist2 / (2.0 * sigma2));
        let phase = kx * dx_src + ky * dy_src - params.source_freq * params.time * 10.0;
        
        // Pulsing source (adds wavepacket periodically)
        let pulse = sin(params.time * params.source_freq * 2.0 * 3.14159);
        if (pulse > 0.9) {
            R_new += envelope * cos(phase) * 0.1;
            I_new += envelope * sin(phase) * 0.1;
        }
    }
    
    // Normalize if enabled (prevents blowup)
    if (params.normalize_waves > 0.5) {
        let mag = sqrt(R_new * R_new + I_new * I_new);
        if (mag > params.wave_range) {
            let scale = params.wave_range / mag;
            R_new *= scale;
            I_new *= scale;
        }
    }
    
    // Store updated wavefunction
    psi_R_next[idx] = R_new;
    psi_I_next[idx] = I_new;
    
    // Create visualization output
    // Encode both real and imaginary parts for custom color rendering
    // We'll use a special encoding: output = R * 1000 + I (scaled appropriately)
    // The render pipeline will decode this
    
    // For now, output probability density |ψ|² scaled, with sign info encoded
    let prob_density = R_new * R_new + I_new * I_new;
    
    // Encode real and imaginary for visualization:
    // Store R in the output, and use the visualization to show both
    // The visualization pipeline will handle the coloring
    output[idx] = R_new * 10.0;  // Scale for visibility
}`;
    }

    getSchrodingerVisualizationShader() {
        // Shader to output visualization data
        return `
// Schrödinger Visualization Shader
// Mode 0-2: Output scalar values for render pipeline color mapping
// Mode 3: Output RGB colors directly (real=red/blue, imaginary=yellow/cyan)

struct VisualizationParams {
    width: u32,
    height: u32,
    time: f32,
    amplitude_scale: f32,
    gamma: f32,
    display_mode: f32,      // 0=real, 1=probability, 2=phase-weighted, 3=real+imag colors, 4=imag only
    prob_scale: f32,        // Scale factor for probability density
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> psi_R: array<f32>;
@group(0) @binding(1) var<storage, read> psi_I: array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: VisualizationParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Get raw wavefunction components
    let R_raw = psi_R[idx];
    let I_raw = psi_I[idx];
    
    // Calculate probability density |ψ|²
    let prob_density = R_raw * R_raw + I_raw * I_raw;
    
    // Scale for visualization
    let amp_scale = params.amplitude_scale;
    let display_mode = params.display_mode;
    
    // Mode 3: Direct RGB color output (bypasses render pipeline color mapping)
    // Mode 4 also handled here for display_mode > 3.5
    if (display_mode > 2.5 && display_mode < 3.5) {
        // Apply user amplitude scale directly
        // Scale down by a constant factor for better dynamic range
        let R = R_raw * amp_scale * 0.05;
        let I = I_raw * amp_scale * 0.05;
        let gamma = params.gamma;
        
        // Use smooth continuous functions instead of hard thresholds
        // Real part: Red (positive) / Blue (negative)
        let R_pos = max(R, 0.0);   // Positive part
        let R_neg = max(-R, 0.0);  // Negative part (as positive value)
        let R_pos_mag = pow(R_pos, gamma);
        let R_neg_mag = pow(R_neg, gamma);
        
        // Imaginary part: Yellow (positive) / Cyan (negative)
        let I_pos = max(I, 0.0);
        let I_neg = max(-I, 0.0);
        let I_pos_mag = pow(I_pos, gamma);
        let I_neg_mag = pow(I_neg, gamma);
        
        // Build color from phase (direction of R and I)
        var phase_color = vec3<f32>(
            R_pos_mag + I_pos_mag * 0.8,                    // Red: positive real + positive imag (yellow)
            I_pos_mag * 0.8 + I_neg_mag * 0.8,             // Green: both yellow and cyan
            R_neg_mag + I_neg_mag * 0.8                     // Blue: negative real + negative imag (cyan)
        );
        
        // Modulate color by amplitude envelope to show probability distribution clearly
        // This makes low-probability regions darker, showing the wavepacket shape
        let amplitude_envelope = sqrt(prob_density) * amp_scale;
        let envelope_factor = pow(amplitude_envelope, 0.4);  // Soft power for visibility
        
        // Apply amplitude modulation to phase colors
        var color = phase_color * envelope_factor;
        
        // Add small base glow for visibility (much reduced)
        let prob_glow = sqrt(prob_density) * params.prob_scale * 0.0001;
        color += vec3<f32>(prob_glow);
        
        // Soft tone mapping: x / (1 + x) - prevents hard clipping to white
        // This maps [0, ∞) → [0, 1) smoothly
        color = color / (vec3<f32>(1.0) + color);
        
        // Boost contrast slightly after tone mapping
        color = pow(color, vec3<f32>(0.9));
        
        // Use alpha=0.5 as a flag to indicate pre-colored data to the render pipeline
        // This is more robust than relying on RGB differences
        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 0.5));
    }
    // Modes 0-2, 4: Output scalar value for render pipeline color mapping
    else {
        var output_value: f32;
        
        // Mode 0: Output real part (will use diverging red-blue colormap)
        if (display_mode < 0.5) {
            output_value = R_raw * amp_scale;
        }
        // Mode 1: Output probability density |ψ|²
        else if (display_mode < 1.5) {
            output_value = prob_density * params.prob_scale;
        }
        // Mode 2: Output magnitude with phase sign encoding
        else if (display_mode < 2.5) {
            let magnitude = sqrt(prob_density);
            let phase = atan2(I_raw, R_raw);
            output_value = magnitude * amp_scale * (cos(phase) + sin(phase) * 0.5);
        }
        // Mode 4: Output imaginary part (debug mode)
        else {
            output_value = I_raw * amp_scale;
        }
        
        // Write scalar value to red channel (render pipeline reads .r channel)
        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(output_value, output_value, output_value, 1.0));
    }
}`;
    }

    getSourceDebugShader() {
        // Debug shader that only shows the source term
        return `
@group(0) @binding(0) var<storage, read> u_prev: array<f32>;
@group(0) @binding(1) var<storage, read> u_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_next: array<f32>;
@group(0) @binding(3) var<storage, read> boundary: array<f32>;
@group(0) @binding(4) var<uniform> params: SimParams;

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
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) {
        return;
    }
    
    let idx = y * width + x;
    
    // Just show the source term directly (no wave equation)
    var source = 0.0;
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dist2 = (f32(x) - center_x) * (f32(x) - center_x) + 
                   (f32(y) - center_y) * (f32(y) - center_y);
        
        let radius = params.source_size;
        let amplitude = params.source_strength * 0.2; // Scaled for visualization
        // Proper frequency scaling: 2π * frequency * time
        let phase = 2.0 * 3.14159265359 * params.source_freq * params.time;
        source = amplitude * sin(phase) * exp(-dist2 / (radius * radius));
    }
    
    // Store source value directly
    u_next[idx] = source;
}`;
    }
}
