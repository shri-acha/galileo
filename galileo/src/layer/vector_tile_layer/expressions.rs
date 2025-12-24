use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Color;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearInterpolationArgs<T> {
    step_values: BTreeSet<StepValue<T>>,
}

impl<T: Clone> LinearInterpolationArgs<T> {
    pub fn new(step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if step_values.len() < 2 {
            return Err("At least 2 step values required".to_string());
        }
        Ok(Self {
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Deserialize)]
pub struct ExponentialInterpolationArgs<T> {
    base: i32,
    step_values: BTreeSet<StepValue<T>>,
}

impl<T: Clone> ExponentialInterpolationArgs<T> {
    pub fn new(base: i32, step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if base <= 0 {
            return Err("Base must be positive".to_string());
        }

        if step_values.len() < 2 {
            return Err("At least 2 step values required".to_string());
        }
        Ok(Self {
            base: base,
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InterpolationArgs<T> {
    Linear(LinearInterpolationArgs<T>),
    Exponential(ExponentialInterpolationArgs<T>),
}

impl<T: Copy> InterpolationArgs<T> {
    fn step_values(&self) -> &BTreeSet<StepValue<T>> {
        match self {
            Self::Linear(args) => &args.step_values,
            Self::Exponential(args) => &args.step_values,
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct StepValue<T> {
    resolution: f64,
    step_value: T,
}

impl<T> PartialEq for StepValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.resolution == other.resolution
    }
}

impl<T> Eq for StepValue<T> {}

impl<T> PartialOrd for StepValue<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.resolution.partial_cmp(&other.resolution)
    }
}

impl<T> Ord for StepValue<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resolution.total_cmp(&other.resolution)
    }
}

/// Type used to define expressions for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateExpression<T> {
    interpolation_type: InterpolationFunction,
    interpolation_args: InterpolationArgs<T>,
}
/// Type used to define steps
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepExpression<T> {
    default_value: T,
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than step resolution
    /// the value T maps to the T value where the step resolution is
    /// less than that of current resolution.
    step_values: BTreeSet<StepValue<T>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResolutionValueRange<T> {
    max_resolution: f64,
    min_resolution: f64,
    start_value: T,
    end_value: T,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValueStep<T> {
    resolution: f64,
    value: T,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InterpolationFunction {
    /// Linear interpolation type with base 1
    Linear,
    /// Exponential interpolation with variable base
    Exponential,
    /// Cubic Bezier interpolation
    Cubic,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum StyleValue<T> {
    /// Simple values like using Color::BLACK
    Simple(T),
    /// Interpolate expression
    Interpolate(InterpolateExpression<T>),
    /// Step expression
    Steps(StepExpression<T>),
}
impl From<Color> for StyleValue<Color> {
    fn from(color_val: Color) -> Self {
        Self::Simple(color_val)
    }
}

impl StyleValue<Color> {
    /// Evaluates value from simple color
    pub fn get_value(&self, current_resolution: f64) -> Color {
        match self {
            StyleValue::Simple(color) => *color,
            StyleValue::Interpolate(expression) => expression.evaluate(current_resolution),
            StyleValue::Steps(expression) => expression.evaluate(current_resolution),
        }
    }
}

impl InterpolateExpression<Color> {
    /// Evaluates value by interpolating color values
    pub fn evaluate(&self, current_resolution: f64) -> Color {
        if let Some(resolution_value_range) = get_resolution_value_range(self, current_resolution) {
            match &self.interpolation_args {
                InterpolationArgs::Linear(_) => {
                    Self::linear_interpolate_color(resolution_value_range, current_resolution)
                }
                InterpolationArgs::Exponential(args) => Self::exponential_interpolate_color(
                    resolution_value_range,
                    args.base,
                    current_resolution,
                ),
            }
        } else {
            self.get_boundary_value(current_resolution)
        }
    }
    fn get_boundary_value(&self, current_resolution: f64) -> Color {
        let step_values = self
            .interpolation_args
            .step_values()
            .iter()
            .collect::<Vec<_>>();

        if current_resolution < step_values[0].resolution {
            step_values[0].step_value
        } else {
            step_values[step_values.len() - 1].step_value
        }
    }

    fn linear_interpolate_color(
        rv_range: ResolutionValueRange<Color>,
        current_resolution: f64,
    ) -> Color {
        Color::rgba(
            linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.r() as f64,
                rv_range.end_value.r() as f64,
                current_resolution,
            ) as u8,
            linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.g() as f64,
                rv_range.end_value.g() as f64,
                current_resolution,
            ) as u8,
            linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.b() as f64,
                rv_range.end_value.b() as f64,
                current_resolution,
            ) as u8,
            linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.a() as f64,
                rv_range.end_value.a() as f64,
                current_resolution,
            ) as u8,
        )
    }

    fn exponential_interpolate_color(
        rv_range: ResolutionValueRange<Color>,
        base: i32,
        current_resolution: f64,
    ) -> Color {
        Color::rgba(
            exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.r() as f64,
                rv_range.end_value.r() as f64,
                current_resolution,
                base,
            ) as u8,
            exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.g() as f64,
                rv_range.end_value.g() as f64,
                current_resolution,
                base,
            ) as u8,
            exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.b() as f64,
                rv_range.end_value.b() as f64,
                current_resolution,
                base,
            ) as u8,
            exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value.a() as f64,
                rv_range.end_value.a() as f64,
                current_resolution,
                base,
            ) as u8,
        )
    }
}

fn exponential_interpolation(
    x_start: f64,
    x_end: f64,
    y_start: f64,
    y_end: f64,
    x0: f64,
    base: i32,
) -> f64 {
    let t: f64 = ((x0 - x_start) / (x_end - x_start).clamp(f64::EPSILON, f64::MAX)).clamp(0.0, 1.0);

    let base: f64 = base as f64;
    let t = if (base - 1.0).abs() > f64::EPSILON {
        (base.powf(t) - 1.0) / (base - 1.0)
    } else {
        t
    };

    let offset = (y_end - y_start) as f64;

    (y_start as f64 + t * (offset)).clamp(0.0, 255.0)
}

fn linear_interpolation(x_start: f64, x_end: f64, y_start: f64, y_end: f64, x0: f64) -> f64 {
    let x_range: f64 = (x_end - x_start).clamp(f64::EPSILON, f64::MAX);

    let k = (y_end - y_start) / x_range;

    let offset = (x0 - x_start).clamp(0.0, x_range);
    return y_start + k * offset;
}

fn get_resolution_value_range<T: Copy>(
    expression: &InterpolateExpression<T>,
    current_resolution: f64,
) -> Option<ResolutionValueRange<T>> {
    let step_values: &BTreeSet<StepValue<T>> = expression.interpolation_args.step_values();
    // Try to find a matching window in step_values
    step_values
        .iter()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| current_resolution >= w[0].resolution && current_resolution <= w[1].resolution)
        .map(|w| ResolutionValueRange {
            min_resolution: w[0].resolution,
            max_resolution: w[1].resolution,
            start_value: w[0].step_value,
            end_value: w[1].step_value,
        })
}

impl<T: Copy> StepExpression<T> {
    pub fn new(default_value: T, step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if step_values.is_empty() {
            return Err("At least 1 step value required".to_string());
        }
        Ok(Self {
            default_value,
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }

    /// Evaluates color value by giving stepwise value
    /// of color on basis of zoom
    pub fn evaluate(&self, current_resolution: f64) -> T {
        if let Some(w) = self
            .step_values
            .iter()
            .collect::<Vec<_>>()
            .windows(2)
            .find(|w| {
                current_resolution >= w[0].resolution && current_resolution <= w[1].resolution
            })
        {
            return w[0].step_value;
        } else if current_resolution < self.step_values.iter().nth(0).unwrap().resolution {
            return self.default_value;
        } else {
            return self
                .step_values
                .iter()
                .nth(self.step_values.len() - 1)
                .unwrap()
                .step_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_resolution_value_range_out_of_bounds() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Linear,
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert!(get_resolution_value_range(&expr, 75.0).is_none());
        assert!(get_resolution_value_range(&expr, 20.0).is_none());
    }

    #[test]
    fn linear_interpolation_bounds() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Linear,
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert_eq!(expr.evaluate(20.0), Color::rgba(0, 0, 0, 0));
        assert_eq!(expr.evaluate(150.0), Color::rgba(128, 128, 128, 128));
    }

    #[test]
    fn linear_interpolation_unordered() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
            StepValue {
                resolution: 0.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
        ])
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Linear,
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert_eq!(expr.evaluate(25.0), Color::rgba(64, 64, 64, 64));
    }

    #[test]
    fn linear_interpolation() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 0.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Linear,
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert_eq!(expr.evaluate(25.0), Color::rgba(64, 64, 64, 64));
    }

    #[test]
    fn exponential_bounds() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 10.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
            ],
        )
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Exponential,
            interpolation_args: InterpolationArgs::Exponential(args),
        };

        assert_eq!(expr.evaluate(5.0), Color::rgba(0, 0, 0, 0));
        assert_eq!(expr.evaluate(150.0), Color::rgba(128, 128, 128, 128));
    }

    #[test]
    fn exponential_interpolation_unordered() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
                StepValue {
                    resolution: 0.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 75.0,
                    step_value: Color::rgba(200, 200, 200, 200),
                },
            ],
        )
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Exponential,
            interpolation_args: InterpolationArgs::Exponential(args),
        };

        assert_eq!(expr.evaluate(25.0), Color::rgba(53, 53, 53, 53));
        assert_eq!(expr.evaluate(60.0), Color::rgba(151, 151, 151, 151));
    }

    #[test]
    fn exponential_interpolation() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 0.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
                StepValue {
                    resolution: 75.0,
                    step_value: Color::rgba(200, 200, 200, 200),
                },
            ],
        )
        .unwrap();

        let expr = InterpolateExpression {
            interpolation_type: InterpolationFunction::Exponential,
            interpolation_args: InterpolationArgs::Exponential(args),
        };

        assert_eq!(expr.evaluate(25.0), Color::rgba(53, 53, 53, 53));
        assert_eq!(expr.evaluate(60.0), Color::rgba(151, 151, 151, 151));
    }

    #[test]
    fn test_step_expression_bounds() {
        let expr = StepExpression::<Color>::new(
            Color::from_hex("#f0f0f0"),
            vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
            ],
        )
        .unwrap();

        assert_eq!(expr.evaluate(5.0), Color::from_hex("#f0f0f0"));
        assert_eq!(expr.evaluate(30.0), Color::from_hex("#1d1d1d"));
    }

    #[test]
    fn test_step_expression() {
        let expr = StepExpression::<Color>::new(
            Color::from_hex("#f0f0f0"),
            vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
                StepValue::<Color> {
                    resolution: 30.0,
                    step_value: Color::from_hex("#1a1a1a"),
                },
            ],
        )
        .unwrap();

        assert_eq!(expr.evaluate(15.0), Color::from_hex("#fafafa"));
        assert_eq!(expr.evaluate(25.0), Color::from_hex("#1d1d1d"));
    }
}
