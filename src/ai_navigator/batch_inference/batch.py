"""Batch inference wrapper for Navigator.

Runs multiple ``chat`` or ``response`` requests sequentially against a
:class:`~ai_navigator.navigator.Navigator` instance.  Designed as a
foundation for future concurrent or async batch processing.

Usage::

    from ai_navigator.navigator import Navigator
    from ai_navigator.batch_inference import BatchInference

    nav = Navigator()
    batch = BatchInference(nav)

    results = batch.chat_batch(
        request_data_list=[
            {"type": "message", "content": "prompt 1"},
            {"type": "message", "content": "prompt 2"},
        ],
        params={"temperature": 0.5},
        configs={"model_name": "my_claude"},
    )
"""
from __future__ import annotations
from typing import Any

from ai_navigator.monitor.logger import get_logger

_log = get_logger("batch_inference")


class BatchInference:
    """Run multiple Navigator requests in sequence.

    Parameters
    ----------
    navigator:
        A :class:`~ai_navigator.navigator.Navigator` instance.
    """

    def __init__(self, navigator: Any) -> None:
        self._navigator = navigator

    def chat_batch(
        self,
        request_data_list: list[dict],
        params: dict | None = None,
        configs: dict | None = None,
    ) -> list[Any]:
        """Run ``navigator.chat`` for each item in *request_data_list*.

        Parameters
        ----------
        request_data_list:
            Each element is a ``request_data`` dict (message / conversation /
            prompt shape) passed directly to ``Navigator.chat``.
        params:
            Forwarded to every ``chat`` call unchanged.
        configs:
            Internal knobs forwarded to every ``chat`` call (must contain
            ``model_name``).

        Returns
        -------
        list
            Responses in the same order as *request_data_list*.
        """
        results = []
        for idx, request_data in enumerate(request_data_list):
            _log.debug("batch chat item %d/%d", idx + 1, len(request_data_list))
            results.append(self._navigator.chat(request_data, params=params, configs=configs))
        return results

    def response_batch(
        self,
        request_data_list: list[dict],
        params: dict | None = None,
        configs: dict | None = None,
    ) -> list[Any]:
        """Run ``navigator.response`` for each item in *request_data_list*.

        Parameters
        ----------
        request_data_list:
            Each element is a ``request_data`` dict (message / conversation /
            prompt shape) passed directly to ``Navigator.response``.
        params:
            Forwarded to every ``response`` call unchanged.
        configs:
            Internal knobs forwarded to every ``response`` call (must contain
            ``model_name``).

        Returns
        -------
        list
            Responses in the same order as *request_data_list*.
        """
        results = []
        for idx, request_data in enumerate(request_data_list):
            _log.debug("batch response item %d/%d", idx + 1, len(request_data_list))
            results.append(self._navigator.response(request_data, params=params, configs=configs))
        return results