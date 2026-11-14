from src.utils import seed_everything, get_device, count_parameters
from src.models import build_model
from src.data import get_supervised_dataloaders, load_class_mapping
from src.training import train_supervised, train_distillation
from src.metrics import evaluate_model, detect_overfitting_epoch
from src.teacher import (
    extract_teacher_features_and_zero_shot,
    evaluate_teacher_zero_shot,
    build_improved_teacher_with_budget,
    build_student_training_records,
)
from src.visualization import (
    plot_training_curves,
    plot_comparison_curves,
    plot_confusion_matrix,
    plot_teacher_noise_diagnostics,
    plot_student_vs_teacher_per_class,
    plot_budget_curve,
    plot_per_class_gain_budget5,
    plot_gradcam_gallery,
)
from src.experiments import (
    run_floor_experiment,
    run_teacher_evaluation,
    run_student_ablation,
    run_budget_curve,
)

__all__ = [
    "seed_everything",
    "get_device",
    "count_parameters",
    "build_model",
    "get_supervised_dataloaders",
    "load_class_mapping",
    "train_supervised",
    "train_distillation",
    "evaluate_model",
    "detect_overfitting_epoch",
    "extract_teacher_features_and_zero_shot",
    "evaluate_teacher_zero_shot",
    "build_improved_teacher_with_budget",
    "build_student_training_records",
    "plot_training_curves",
    "plot_comparison_curves",
    "plot_confusion_matrix",
    "plot_teacher_noise_diagnostics",
    "plot_student_vs_teacher_per_class",
    "plot_budget_curve",
    "plot_per_class_gain_budget5",
    "plot_gradcam_gallery",
    "run_floor_experiment",
    "run_teacher_evaluation",
    "run_student_ablation",
    "run_budget_curve",
]
