# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""retriever_Estimator primitive."""

from __future__ import annotations
import json
import time
import sys
import itertools
from typing import Optional, Dict, Union, Iterable
import logging

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from qiskit_ibm_runtime.runtime_job import RuntimeJob
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_ibm_runtime.options import Options
from qiskit_ibm_runtime.options.estimator_options import EstimatorOptions
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder
from qiskit_ibm_runtime.base_primitive import BasePrimitiveV1, BasePrimitiveV2
from qiskit_ibm_runtime.utils.deprecation import deprecate_arguments, issue_deprecation_msg

# pylint: disable=unused-import,cyclic-import
from qiskit_ibm_runtime.session import Session
from qiskit_ibm_runtime.batch import Batch

logger = logging.getLogger(__name__)


class Estimator:
    """Base class for Qiskit Runtime Estimator."""

    version = 0

class RetrieverEstimatorV2(BasePrimitiveV2[EstimatorOptions], Estimator, BaseEstimatorV2):
    """Class for interacting with Qiskit Runtime Estimator primitive service.

    Qiskit Runtime Estimator primitive service estimates expectation values of quantum circuits and
    observables. """
    
    _options_class = EstimatorOptions

    version = 0

    def __init__(
            self,
            mode: Optional[Union[BackendV1, BackendV2, Session, Batch, str]] = None,
            backend: Optional[Union[str, BackendV1, BackendV2]] = None,
            session: Optional[Session] = None,
            options: Optional[Union[Dict, EstimatorOptions]] = None,
        ):
        """Initializes the Estimator primitive.
    
            Args:
                mode: The execution mode used to make the primitive query. It can be:"""
        BaseEstimatorV2.__init__(self)
        Estimator.__init__(self)
        self._options=options
        print("RetrieverEstimatorV2 - for IBM Quantum Challenge 2024")
        
    def run(
            self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None,
        ):
    
        """Submit a request to the estimator primitive.
            Args:
            Returns:
            """
        self._animation(text=" ")
        print("These are pre-run results from a previous quantum system run" + "\n")
    
        file_prefix = self.options['path']
        result_file = f"{file_prefix}_result.json"
        inputs_file = f"{file_prefix}_inputs.json"
        
        self._animation(text="Loading results from json...")
        
        saved_result = self._load_json(result_file)
        self._load_json(inputs_file)
        pub_result = saved_result[0]
        self._animation(text="Storing results...")

        print("\n" + "Results loaded! ")
        return pub_result
        
    def _animation(self, text='', duration=1.5, delay=0.1, bar_length=30):
        animation = itertools.cycle(['|', '/', '-', '\\'])
        time.time() + duration
        total_steps = int(duration / delay)
        
        for i in range(total_steps):
            current_frame = next(animation)
            progress = (i + 1) / total_steps
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            progress_percent = int(progress * 100)
            
            sys.stdout.write(f'\r{text} {current_frame} [{bar}] {progress_percent}%')
            sys.stdout.flush()
            time.sleep(delay)
        
        sys.stdout.write(f'\r{text} Done! [{"=" * bar_length}] 100%\n')
      
    def _load_json(self, filename: str) -> any:
        """Load JSON from a file"""
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file, cls=RuntimeDecoder)
        return data

    @classmethod
    def _program_id(cls) -> str:
        """Return the program ID."""
        return "estimator"

    def _validate_options(self, options: dict) -> None:
        """Validate that primitive inputs (options) are valid

        Raises:
            ValidationError: if validation fails.
            ValueError: if validation fails.
        """
        try:
            options['path']
        except KeyError: f"Enter a file path in options"
        except ValueError: f"Enter a valid file path"
        Options.validate_options(options)