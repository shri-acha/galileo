use serde::{Deserialize, Serialize};

use crate::error::GalileoError;
use crate::Color;

/// Arguments for exponential interpolation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearArgs<T> {
    step_values: Vec<StepValue<T>>,
}

/// Generic arguments for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolationArgs<T> {
    base: Option<i32>,
    step_values: Option<Vec<StepValue<T>>>,
}
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepValue<T> {
    resolution: f64,
    step_value: T,
}
/// Context for the interpolation
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateContext {
    pub current_resolution: f64,
}

/// Type used to define expressions for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateExpression<T> {
    start_value: T,
    end_value: T,
    max_resolution: Option<f64>,
    min_resolution: Option<f64>,
    interpolation_type: Interpolation,
    interpolation_args: InterpolationArgs<T>,
}
/// Type used to define steps
#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct StepExpression<T: Ord> {
    default_value: T,
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than step resolution
    /// the value T maps to the T value where the step resolution is
    /// less than that of current resolution.
    step_resolution: Vec<f64>,
    step_value_: Vec<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct ResolutionValueRange<T> {
    max_resolution: f64,
    min_resolution: f64,
    start_value: T,
    end_value: T,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialOrd, PartialEq)]
pub struct ValueStep<T> {
    resolution: f64,
    value: T,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Interpolation {
    /// Linear interpolation type with base 1
    Linear,
    /// Exponential interpolation with variable base
    Exponential,
    /// Cubic Bezier interpolation
    Cubic,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum StyleValue<T: Ord> {
    /// Simple values like using Color::BLACK
    Simple(T),
    /// Interpolate expression
    Interpolate(InterpolateExpression<T>),
    /// Step expression
    Steps(StepExpression<T>),
}

impl Color {
    /// Evaluates value from simple color
    pub fn get_value(&self, _: InterpolateContext) -> Result<Color, GalileoError> {
        Ok(*self)
    }
}

impl InterpolateExpression<Color> {
    /// Evaluates value by interpolating color values
    pub fn get_value(&self, current_resolution: f64) -> Option<Color> {
        let resolution_value_range: Option<ResolutionValueRange<Color>> =
            self.get_resolution_value_range(current_resolution);

        if let Some(resolution_value_range) = resolution_value_range {
            match self.interpolation_type {
                Interpolation::Linear => {
                    let interpolated_color: Color = Self::lerp(
                        resolution_value_range.start_value,
                        resolution_value_range.end_value,
                        &self.interpolation_args,
                        Some(resolution_value_range.min_resolution),
                        Some(resolution_value_range.max_resolution),
                        current_resolution,
                    )?;
                    Some(interpolated_color)
                }
                Interpolation::Exponential => {
                    let interpolated_color: Color = Self::exponential_interpolate(
                        self.start_value,
                        self.end_value,
                        self.min_resolution,
                        self.max_resolution,
                        &self.interpolation_args,
                        current_resolution,
                    )?;
                    Some(interpolated_color)
                }
                Interpolation::Cubic => {
                    todo!()
                }
            }
        } else {
            None
        }
    }

    fn lerp(
        start_value: Color,
        end_value: Color,
        interpolation_args: &InterpolationArgs<Color>,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        current_resolution: f64,
    ) -> Option<Color> {
        match (max_resolution, min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                let resolution_range: f64 =
                    (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX);
                // individual ratios for each field
                let kr = (end_value.r() - start_value.r()) as f64 / resolution_range;
                let kg = (end_value.g() - start_value.g()) as f64 / resolution_range;
                let kb = (end_value.b() - start_value.b()) as f64 / resolution_range;
                let ka = (end_value.a() - start_value.a()) as f64 / resolution_range;

                let offset = (current_resolution - min_resolution).clamp(0.0, resolution_range);

                Some(Color::rgba(
                    (start_value.r() as f64 + kr * offset).clamp(0.0, 255.0) as u8,
                    (start_value.g() as f64 + kg * offset).clamp(0.0, 255.0) as u8,
                    (start_value.b() as f64 + kb * offset).clamp(0.0, 255.0) as u8,
                    (start_value.a() as f64 + ka * offset).clamp(0.0, 255.0) as u8,
                ))
            }
            (_, _) => None,
        }
    }

    fn exponential_interpolate(
        start_value: Color,
        end_value: Color,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        interpolation_args: &InterpolationArgs<Color>,
        current_resolution: f64,
    ) -> Option<Color> {
        match (max_resolution, min_resolution) {
            (Some(max_resolution), Some(min_resolution)) => {
                let t: f64 = ((current_resolution - min_resolution)
                    / (max_resolution - min_resolution).clamp(f64::EPSILON, f64::MAX))
                .clamp(0.0, 1.0);

                if let Some(base) = interpolation_args.base {
                    let base: f64 = base as f64;
                    let t = if (base - 1.0).abs() > f64::EPSILON {
                        (base.powf(t) - 1.0) / (base - 1.0)
                    } else {
                        t
                    };

                    let offset_r = (end_value.r() - start_value.r()) as f64;
                    let offset_g = (end_value.g() - start_value.g()) as f64;
                    let offset_b = (end_value.b() - start_value.b()) as f64;
                    let offset_a = (end_value.a() - start_value.a()) as f64;

                    Some(Color::rgba(
                        (start_value.r() as f64 + t * (offset_r)).clamp(0.0, 255.0) as u8,
                        (start_value.g() as f64 + t * (offset_g)).clamp(0.0, 255.0) as u8,
                        (start_value.b() as f64 + t * (offset_b)).clamp(0.0, 255.0) as u8,
                        (start_value.a() as f64 + t * (offset_a)).clamp(0.0, 255.0) as u8,
                    ))
                } else {
                    None
                }
            }
            (_, _) => None,
        }
    }

    fn cubic_interpolate(
        step_value: Vec<&StepValue<Color>>,
        min_resolution: Option<f64>,
        max_resolution: Option<f64>,
        interpolation_args: &InterpolationArgs<Color>,
        current_resolution: f64,
    ) -> Option<Color> {
        todo!();
    }
    fn get_resolution_value_range(
        &self,
        current_resolution: f64,
    ) -> Option<ResolutionValueRange<Color>> {
        self.interpolation_args
            .step_values
            .as_ref()?
            .windows(2)
            .find(|w| {
                current_resolution >= w[0].resolution && current_resolution <= w[1].resolution
            })
            .map(|w| ResolutionValueRange {
                min_resolution: w[0].resolution,
                max_resolution: w[1].resolution,
                start_value: w[0].step_value,
                end_value: w[1].step_value,
            })
    }
}

impl StepExpression<Color> {
    /// Evaluates color value by giving stepwise value
    /// of color on basis of zoom
    pub fn get_value(&self, _resolution: f64) -> Option<Color> {
        Some(Color::BLACK)
    }
}
