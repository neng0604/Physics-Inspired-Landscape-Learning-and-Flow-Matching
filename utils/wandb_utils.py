import os
from typing import Any, Dict, Optional


def _convert_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _convert_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_config(v) for v in value]
    if hasattr(value, 'items'):
        return {str(k): _convert_config(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def maybe_init_wandb(
    *,
    logger,
    enabled: bool,
    project: Optional[str],
    entity: Optional[str],
    run_name: Optional[str],
    config: Any,
    log_dir: str,
    tags: Optional[list] = None,
    mode: Optional[str] = None,
):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning('wandb is not installed; continuing without wandb logging.')
        return None

    if project is None:
        project = os.environ.get('WANDB_PROJECT', 'PAFlow')
    if entity is None:
        entity = os.environ.get('WANDB_ENTITY')
    if mode is None:
        mode = os.environ.get('WANDB_MODE')

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=_convert_config(config),
        dir=log_dir,
        tags=tags,
        mode=mode,
    )
    logger.info(
        'Initialized wandb run: project=%s entity=%s name=%s mode=%s',
        project,
        entity,
        run.name if run is not None else run_name,
        mode,
    )
    return run


def log_metrics(run, metrics: Dict[str, Any], step: Optional[int] = None):
    if run is None:
        return
    clean_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            try:
                clean_metrics[key] = value.item()
                continue
            except Exception:
                pass
        if isinstance(value, (int, float, bool)):
            clean_metrics[key] = value
    if clean_metrics:
        run.log(clean_metrics, step=step)


def log_figure(run, key: str, figure, step: Optional[int] = None):
    if run is None:
        return
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    run.log({key: wandb.Image(figure)}, step=step)


def finish(run):
    if run is None:
        return
    run.finish()
