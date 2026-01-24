use crate::egui_tools::EguiRenderer;
use bytemuck::bytes_of;
use egui_wgpu::{ScreenDescriptor, wgpu::SurfaceError};
use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use winit::event::{DeviceEvent, MouseButton};
use std::num::NonZero;
use std::{borrow::Cow, f32::consts::FRAC_PI_2};
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorWrites, ComputePipeline, ComputePipelineDescriptor, Device, Extent3d, FilterMode,
    FragmentState, FrontFace, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, Surface,
    SurfaceConfiguration, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent, event_loop::ActiveEventLoop, keyboard::{PhysicalKey, KeyCode}, window::{CursorGrabMode, Window, WindowId}};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { pos: Default::default(), pitch: Default::default(), yaw: Default::default(), fov: FRAC_PI_2 }
    }
}
impl Camera {
    pub fn write_uniform(&self, uniform: &mut [u8]) {
        let mat = Mat4::from_translation(self.pos) * Mat4::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        uniform[0..64].copy_from_slice(bytes_of(&mat));
        uniform[64..68].copy_from_slice(bytes_of(&self.fov));
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PType {
    Cube = 1,
    Sphere = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct Primitive {
    pub ty: PType,
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
}

#[derive(Debug, Default, Clone)]
pub struct Scene {
    pub primitives: Vec<Primitive>,
}
impl Scene {
    pub fn write_uniform(&self, uniform: &mut [u8]) {
        uniform[68..72].copy_from_slice(&(self.primitives.len() as u32).to_le_bytes());
        for (primitive, i) in self.primitives.iter().zip((0..).map(|x| x * 144 + 256)) {
            uniform[i..i + 4].copy_from_slice(&(primitive.ty as u32).to_le_bytes());
            let transform = Mat4::from_translation(primitive.pos) * Mat4::from_quat(primitive.rot) * Mat4::from_scale(primitive.scale);
            uniform[i+16..i+80].copy_from_slice(bytes_of(&transform));
            uniform[i+80..i+144].copy_from_slice(bytes_of(&transform.inverse()));
        }
    }
}

pub struct AppState {
    pub keys: HashSet<PhysicalKey>,
    pub camera: Camera,
    pub scene: Scene,
    pub device: Device,
    pub queue: Queue,
    pub surface_config: SurfaceConfiguration,
    pub surface: Surface<'static>,
    pub texture: Texture,
    pub sampler: Sampler,
    pub uniform_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub compute_bind_group_layout: BindGroupLayout,
    pub compute_bind_group: BindGroup,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub pipeline: RenderPipeline,

    pub focused_renderer: bool,
    pub scale_factor: f32,
    pub egui_renderer: EguiRenderer,
}

impl AppState {
    async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        window: &Window,
        width: u32,
        height: u32,
    ) -> Self {
        let power_pref = wgpu::PowerPreference::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let features = wgpu::Features::CLEAR_TEXTURE;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let selected_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swapchain_format = swapchain_capabilities
            .formats
            .iter()
            .find(|d| **d == selected_format)
            .expect("failed to select proper surface texture format!");

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: *swapchain_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 0,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &surface_config);

        let texture = Self::new_texture(&device, width, height);
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 2048,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                    },
                ],
            });
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniform_buffer, offset: 0, size: NonZero::new(256) }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniform_buffer, offset: 256, size: None }),
                },
            ],
        });
        let compute = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
        });
        let desc = PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        };
        let layout = device.create_pipeline_layout(&desc);
        let desc = ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &compute,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        };
        let compute_pipeline = device.create_compute_pipeline(&desc);

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        let vertex = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("vertex.wgsl"))),
        });
        let fragment = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("fragment.wgsl"))),
        });

        let desc = PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(&desc);
        let desc = RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &vertex,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &fragment,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: *swapchain_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        };
        let pipeline = device.create_render_pipeline(&desc);

        let egui_renderer = EguiRenderer::new(&device, surface_config.format, 1, window);

        let scale_factor = 1.0;

        let scene = Scene {
            primitives: vec![
                Primitive { ty: PType::Cube, pos: Vec3::ZERO, rot: Quat::IDENTITY, scale: Vec3::ONE },
                Primitive { ty: PType::Sphere, pos: Vec3::new(0.0, 0.0, 4.0), rot: Quat::IDENTITY, scale: Vec3::ONE },
            ]
        };

        Self {
            keys: Default::default(),
            camera: Default::default(),
            scene,
            device,
            queue,
            surface,
            surface_config,
            texture,
            sampler,
            uniform_buffer,
            compute_pipeline,
            compute_bind_group_layout,
            compute_bind_group,
            bind_group_layout,
            bind_group,
            pipeline,
            focused_renderer: false,
            egui_renderer,
            scale_factor,
        }
    }

    fn new_texture(device: &Device, width: u32, height: u32) -> Texture {
        let descriptor = TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        device.create_texture(&descriptor)
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }
}

pub struct App {
    instance: wgpu::Instance,
    state: Option<AppState>,
    window: Option<Arc<Window>>,
}

impl App {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        Self {
            instance,
            state: None,
            window: None,
        }
    }

    async fn set_window(&mut self, window: Window) {
        let window = Arc::new(window);
        let initial_width = 1360;
        let initial_height = 768;

        let _ = window.request_inner_size(PhysicalSize::new(initial_width, initial_height));

        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("Failed to create surface!");

        let state = AppState::new(
            &self.instance,
            surface,
            &window,
            initial_width,
            initial_width,
        )
        .await;

        self.window.get_or_insert(window);
        self.state.get_or_insert(state);
    }

    fn handle_resized(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let state = self.state.as_mut().unwrap();
            state.resize_surface(width, height);
            state.texture = AppState::new_texture(&state.device, width, height);
            let texture_view = state.texture.create_view(&TextureViewDescriptor::default());

            state.bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &state.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&state.sampler),
                    },
                ],
            });
            state.compute_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &state.compute_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(wgpu::BufferBinding { buffer: &state.uniform_buffer, offset: 0, size: NonZero::new(256) }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(wgpu::BufferBinding { buffer: &state.uniform_buffer, offset: 256, size: None }),
                    },
                ],
            });
        }
    }

    fn handle_redraw(&mut self) {
        let state = self.state.as_mut().unwrap();

        if state.keys.contains(&PhysicalKey::Code(KeyCode::ArrowLeft)) {
            state.camera.yaw -= 0.05;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::ArrowRight)) {
            state.camera.yaw += 0.05;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::ArrowUp)) {
            state.camera.pitch -= 0.05;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::ArrowDown)) {
            state.camera.pitch += 0.05;
        }
        state.camera.pitch = state.camera.pitch.clamp(-FRAC_PI_2, FRAC_PI_2);

        let mut movement = Vec2::ZERO;
        if state.keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) {
            movement += Vec2::Y;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) {
            movement += Vec2::NEG_Y;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) {
            movement += Vec2::X;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) {
            movement += Vec2::NEG_X;
        }
        movement = Vec2::from_angle(-state.camera.yaw).rotate(movement);
        movement /= 10.0;
        state.camera.pos += Vec3::new(movement.x, 0.0, movement.y);

        if state.keys.contains(&PhysicalKey::Code(KeyCode::Space)) {
            state.camera.pos += Vec3::Y * 0.05;
        }
        if state.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft)) {
            state.camera.pos += Vec3::Y * -0.05;
        }

        state.scene.primitives[0].rot *= Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.01);

        // Attempt to handle minimizing window
        if let Some(window) = self.window.as_ref()
            && let Some(true) = window.is_minimized()
        {
            println!("Window is minimized");
            return;
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [state.surface_config.width, state.surface_config.height],
            pixels_per_point: self.window.as_ref().unwrap().scale_factor() as f32
                * state.scale_factor,
        };

        let surface_texture = state.surface.get_current_texture();

        match surface_texture {
            Err(SurfaceError::Outdated) => {
                // Ignoring outdated to allow resizing and minimization
                println!("wgpu surface outdated");
                return;
            }
            Err(_) => {
                surface_texture.expect("Failed to acquire next swap chain texture");
                return;
            }
            Ok(_) => {}
        };

        let surface_texture = surface_texture.unwrap();

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut data = [0; 2048];
        state.camera.write_uniform(&mut data);
        state.scene.write_uniform(&mut data);
        state.queue.write_buffer(&state.uniform_buffer, 0, &data);

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_bind_group(0, Some(&state.compute_bind_group), &[]);
            compute_pass.set_pipeline(&state.compute_pipeline);
            compute_pass.dispatch_workgroups(
                state.texture.width().div_ceil(16),
                state.texture.height().div_ceil(16),
                1,
            );
        }

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main scene"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_bind_group(0, Some(&state.bind_group), &[]);
            render_pass.set_pipeline(&state.pipeline);
            render_pass.draw(0..4, 0..1);
        }

        let window = self.window.as_ref().unwrap();

        {
            state.egui_renderer.begin_frame(window);

            egui::Window::new("egui window")
                .resizable(true)
                .vscroll(true)
                .default_open(false)
                .show(state.egui_renderer.context(), |ui| {
                    ui.label("label");

                    if ui.button("button").clicked() {
                        println!("click")
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Pixels per point: {}",
                            state.egui_renderer.context().pixels_per_point()
                        ));
                        if ui.button("-").clicked() {
                            state.scale_factor = (state.scale_factor - 0.1).max(0.3);
                        }
                        if ui.button("+").clicked() {
                            state.scale_factor = (state.scale_factor + 0.1).min(3.0);
                        }
                    });
                });

            state.egui_renderer.end_frame_and_draw(
                &state.device,
                &state.queue,
                &mut encoder,
                window,
                &surface_view,
                screen_descriptor,
            );
        }

        state.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        pollster::block_on(self.set_window(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // let egui render to process the event first
        self.state
            .as_mut()
            .unwrap()
            .egui_renderer
            .handle_input(self.window.as_ref().unwrap(), &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw();

                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                self.handle_resized(new_size.width, new_size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !self.state.as_mut().unwrap().egui_renderer.context().wants_keyboard_input() {
                    match event.state {
                        winit::event::ElementState::Pressed => {
                            self.state.as_mut().unwrap().keys.insert(event.physical_key);
                            if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                                self.state.as_mut().unwrap().focused_renderer = false;
                            }
                        }
                        winit::event::ElementState::Released => { self.state.as_mut().unwrap().keys.remove(&event.physical_key); }
                    }
                }
            }
            WindowEvent::MouseInput { button, .. } => {
                if button == MouseButton::Left {
                    self.state.as_mut().unwrap().focused_renderer = true;
                }
            }
            _ => (),
        }
        let state = self.state
            .as_mut()
            .unwrap();
        state.focused_renderer &= !state.egui_renderer.context().wants_pointer_input();

        let window = self.window.as_mut().unwrap();
        window.set_cursor_visible(!state.focused_renderer);
        let _ = window.set_cursor_grab(
            if state.focused_renderer {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            }
        );
        if !state.focused_renderer {
            state.keys.clear();
        }
    }

    fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: winit::event::DeviceEvent,
        ) {
        let state = self.state.as_mut().unwrap();
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event && state.focused_renderer {
               state.camera.yaw += x as f32 / 768.0;
               state.camera.pitch += y as f32 / 768.0;
        }
    }
}
