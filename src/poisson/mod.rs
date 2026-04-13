use glam::Vec3;
use rand::{SeedableRng, rngs::SmallRng};

use crate::app::PortalPair;

mod sphere_sampling;

#[derive(Debug)]
pub struct Simulation {
    points: Vec<Vec3>,
}
impl Simulation {
    pub fn new(min: Vec3, max: Vec3, _portals: &[PortalPair]) -> Self {
        let points = sphere_sampling::sample_points(&mut SmallRng::from_os_rng(), min, max);
        Self {
            points,
        }
    }
}
