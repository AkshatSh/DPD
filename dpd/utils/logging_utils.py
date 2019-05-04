from typing import (
    List,
    Tuple,
    Dict,
)

import logging

from .logger import Logger


MetricsType = Dict[str, object]

def log_train_metrics(
    logger: Logger,
    metrics: MetricsType,
    step: int,
    prefix='al'
):
    def log_special_metrics(metric_name: str, metric_val: object) -> List[Tuple[str, int]]:
        if type(metric_val) == int or type(metric_val) == float:
            return [(metric_name, metric_val)]
        elif type(metric_val) == list:
            res = []
            for metric_val_item in metric_val:
                class_label = metric_val_item['class']
                for metric_n, metric_v in metric_val_item.items():
                    if metric_n == 'class':
                        # skip class names
                        continue
                    res.append(
                        (
                            f'{metric_name}_{class_label}_{metric_n}',
                            metric_v,
                        )
                    )
            return res
        else:
            logging.warning(f'Unknown metric type: {type(metric_val)} for ({metric_name}, {metric_val})')
            return []

    metric_list = []
    for metric, val in metrics.items():
        metric_name = metric
        set_name = 'train'
        if metric.startswith('best_validation'):
            set_name = 'valid'
            metric_name = metric[len('best_validation_'):]
        elif metric.startswith('training'):
            set_name = 'train'
            metric_name = metric[len('training_'):]
        else:
            # ignore any other metric types
            continue
        
        if metric_name.startswith('_'):
            # ignore hidden
            metric_name = metric_name[1:]
        
        full_metric_name = f'{prefix}/{set_name}/{metric_name}'
        
        metric_list.extend(
            filter(
                lambda x: x is not None,
                log_special_metrics(
                    metric_name=full_metric_name,
                    metric_val=val,
                ),
            ),
        )

    for metric_name, metric_val in metric_list:
        logger.scalar_summary(tag=metric_name, value=metric_val, step=step)