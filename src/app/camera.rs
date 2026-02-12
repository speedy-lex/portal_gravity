use std::{f32::consts::FRAC_PI_2, ops::Mul};

use bytemuck::bytes_of;
use glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles};

use crate::app::PortalPair;

#[derive(Debug, Clone, Copy)]
struct Segment {
    pub start: Vec3,
    pub end: Vec3,
}
impl Segment {
    fn intersect_disk(&self, inv_transform: Mat4) -> bool {
        let transformed = inv_transform * *self;
        let orig = transformed.start;
        let dir = transformed.end - transformed.start;

        if dir.z == 0.0 {
            return false;
        }

        let t = -orig.z / dir.z;
        if !(0.0..=1.0).contains(&t) {
            return false;
        }

        let pos = transformed.start.lerp(transformed.end, t);
        pos.length_squared() <= 1.0
    }
}
impl Mul<Segment> for Mat4 {
    type Output = Segment;

    fn mul(self, rhs: Segment) -> Self::Output {
        let start = self * Vec4::new(rhs.start.x, rhs.start.y, rhs.start.z, 1.0);
        let end = self * Vec4::new(rhs.end.x, rhs.end.y, rhs.end.z, 1.0);
        Segment {
            start: start.xyz() / start.w,
            end: end.xyz() / end.w,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub up: Vec3,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Default::default(),
            rot: Quat::IDENTITY,
            up: Vec3::Y,
            fov: FRAC_PI_2,
        }
    }
}
impl Camera {
    pub fn write_uniform(&self, uniform: &mut [u8]) {
        let mat = Mat4::from_translation(self.pos) * Mat4::from_quat(self.rot);
        uniform[0..64].copy_from_slice(bytes_of(&mat));
        uniform[64..68].copy_from_slice(bytes_of(&self.fov));
    }
    pub fn update(&mut self, movement: Vec3, portals: &[PortalPair]) {
        let segment = Segment {
            start: self.pos,
            end: self.pos + movement,
        };

        for portal_pair in portals {
            let (transform_a, transform_b) = portal_pair.get_transforms();

            if segment.intersect_disk(transform_a.inverse()) {
                let transform = transform_b * transform_a.inverse();
                self.pos = (transform * segment).end;
                self.rot = transform.to_scale_rotation_translation().1 * self.rot;
                self.up = (transform * Vec4::new(self.up.x, self.up.y, self.up.z, 0.0)).xyz();
                return;
            }
            if segment.intersect_disk(transform_b.inverse()) {
                let transform = transform_a * transform_b.inverse();
                self.pos = (transform * segment).end;
                self.rot = transform.to_scale_rotation_translation().1 * self.rot;
                self.up = (transform * Vec4::new(self.up.x, self.up.y, self.up.z, 0.0)).xyz();
                return;
            }
        }

        self.pos = segment.end;
    }
}
