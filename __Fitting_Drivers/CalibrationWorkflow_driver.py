import Simulation_driver as SM
import Surrogate_driver_sklearn as SD
from Scaler_driver import DataScaler as DS
import PSO_driver_withscaler as PSO
from Online_learning_driver import OnlineLearning

import pandas as pd
import numpy as np
import time
from sklearn.neural_network import MLPRegressor
from enum import Enum
import logging

class CalibrationType(Enum):
    DIRECT = 'direct'
    SURROGATE = 'surrogate'
    ONLINE = 'online'

class CalibrationWorkflow:
    def __init__(self, process_data, workflow_settings = None, calibration_settings=None):
        self.process_data = process_data
        self.simulation_model_settings = None  # This will be set later
        self.simulation_model = None
        self.surrogate_model_settings = None
        self.surrogate_model = None  # This will be set later
        self.online_learner = None  # Placeholder for online learning
        self.calibration_settings = calibration_settings
        self.workflow_settings = workflow_settings
        self.calibration_results = None
        self.simulation_data_collect = None  # Placeholder for simulation data collection

        self.next_data_item = None  # For online learning validation purpose
        
        calibration_type_str = workflow_settings.get('calibration_type', 'direct') if workflow_settings else 'direct'
        if calibration_type_str not in CalibrationType._value2member_map_:
            raise ValueError(f"Invalid calibration_type: {calibration_type_str}. Must be one of {[e.value for e in CalibrationType]}")
        self.calibration_type = CalibrationType(calibration_type_str)
        
        self.run_direct_calib_for_not_accurate = workflow_settings.get('run_direct_for_not_accurate', False) if workflow_settings else False
        logging_level_str = (workflow_settings or {}).get('logging_level', 'INFO')
        logging_level = getattr(logging, logging_level_str.upper(), logging.INFO)

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s: %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging_level)
        self.logger.propagate = False
        self.logger.debug("Logger initialized.")
    
    def log_milestone(self, message):
        self.logger.info("\n" + "="*60)
        self.logger.info(message.upper())
        self.logger.info("="*60 + "\n")

    def log_minor_milestone(self, message):
        """Highlight minor workflow steps in the log."""
        self.logger.info("-" * 40)
        self.logger.info(message)
        self.logger.info("-" * 40)

    def set_simulation_model(self, simulation_model_setting):
        self.simulation_model_settings = simulation_model_setting 
        # {'hy_filename' :  '../00_Modell_database/Simulation/SampleModel_V2.hsc','cols_mapping' : ..., 'resultindict': True}
    
    def set_surrogate_model(self, surrogate_model_settings):
        self.surrogate_model_settings = surrogate_model_settings
        # __init__(self, x_cols=None, y_cols=None, model=None, scaler=None, load_path=None):

    def check_models(self):
        if self.simulation_model is not None and self.surrogate_model is not None:
            if set(self.simulation_model.x_cols) != set(self.surrogate_model.x_cols):
                raise KeyError("Input columns of simulation and surrogate model do not match!")
            if set(self.simulation_model.y_cols) != set(self.surrogate_model.y_cols):
                raise KeyError("Output columns of simulation and surrogate model do not match!")
            if self.surrogate_model.scaler is None:
                raise ImportError("Surrogate model scaler is not set. It is mandatory to set a scaler for the surrogate model to ensure proper data scaling during calibration.")
            if self.surrogate_model.model is None:
                raise ImportError("Surrogate model is not set. It is mandatory to set a surrogate model to ensure proper calibration.")
    
    def init_models(self):
        if self.simulation_model_settings is not None:
            x_cols = list(self.simulation_model_settings['cols_mapping']['InputParams'].keys())
            y_cols = list(self.simulation_model_settings['cols_mapping']['OutputParams'].keys())
            simres = {param: None for param in y_cols}

            self.simulation_model = SM.simulation_driver(
                hy_filename = self.simulation_model_settings['hy_filename'], 
                cols_mapping = self.simulation_model_settings['cols_mapping'], 
                x_cols = x_cols, 
                y_cols = y_cols, 
                resultindict=self.simulation_model_settings.get('resultindict',True), 
                incaseofnooutput = simres)
            self.simulation_model.load_model()
            self.logger.info(f"Simulation model is initialized, and model loaded from: {self.simulation_model.HyCase.Title.Value}")
        else:
             raise ValueError("Simulation model is not set.") # It is mandatory to set a simulation model as it is the main driver of the calibration workflow.

        # This is optional
        if self.surrogate_model_settings is not None:
            # Load model if already existing
            if 'load_path' in self.surrogate_model_settings:
                self.surrogate_model = SD.surrogate_driver(load_path = self.surrogate_model_settings['load_path'])  
                self.logger.info(f"Surrogate model is initialized from: {self.surrogate_model_settings['load_path']}")
                self.check_models()  
            else:
                raise Warning("Surrogate model load path is not set. A new model will be loaded. Currently only sklearn models are supported! And the model should be trained and saved before using it in the calibration workflow.")
        else:
            raise Warning("Surrogate model is not set. It will be not used!")
    
    def init_online_learning(self, training_data, **kwargs):
        if self.surrogate_model is not None:
            # Initialize online learner with the surrogate model
            batch_size = kwargs.get('batch_size', 1)  # Default batch size
            replay_ratio = kwargs.get('replay_ratio', 0.01)
            retrain_scaler = kwargs.get('retrain_scaler', False)

            if len(training_data) == 0:
                raise ValueError("Training data is empty. Please provide the training dataset of original surrogate model.")
            y_fitt = self.calibration_settings.get('y_fitt') or {param: {'Weight': 1, 'SDEV': 1} for param in self.surrogate_model.y_cols}

            self.online_learner = OnlineLearning(
                surrogate_model=self.surrogate_model,  # your surrogate_driver object
                original_data=training_data,  # your original training dataset
                y_fitt=y_fitt,
                batch_size=batch_size,  # retrain every x outliers
                replay_ratio=replay_ratio,  # use x ratio of original data in retraining
                retrain_scaler=retrain_scaler,  # set to True if you want to retrain scaler TODO: it should be finisihed
                logger=self.logger  # pass the logger for online learning
            )
            self.logger.info(f"Online learning initialized with surrogate model and settings : {kwargs}")
        else:
            raise ValueError("Surrogate model is not set. It is mandatory to set a surrogate model for online learning in the calibration workflow.")

    @staticmethod
    def create_particles_with_full_state(item, x_cols, particles_for_fittparams):
        """
        Create a dictionary of particles with full state.
        Each particle is a dict with all x_cols parameters set from item,
        except those in particles_for_fittparams, which are set from the corresponding value.
        
        Args:
            item (pd.Series): Row from DataFrame with base values.
            x_cols (list): List of all parameter names.
            particles_for_fittparams (dict): Dict of {param: np.array of values for each particle}.
        
        Returns:
            dict: {particle_index: {param: value, ...}, ...}
        """
        # Number of particles is determined by the length of any value array in particles_for_fittparams
        particle_num = len(next(iter(particles_for_fittparams.values())))
        particles = {}
        for i in range(particle_num):
            particle = {}
            for param in x_cols:
                if param in particles_for_fittparams:
                    particle[param] = particles_for_fittparams[param][i]
                else:
                    particle[param] = item[param]
            particles[i+1] = particle  # 1-based indexing
        return particles

    def check_accuracy(self, sim_validation_cost, sim_validation_treshold=None, debug_workflow = False):
        if sim_validation_cost is not None and sim_validation_cost <= sim_validation_treshold:
            workingpoint_accurate = True
            self.logger.info(f"Working point is accurate enough with cost: {sim_validation_cost:.6f} < threshold {sim_validation_treshold:.6f}")
        else:
            workingpoint_accurate = False
            self.logger.info(f"Working point is NOT accurate enough with cost: {sim_validation_cost:.6f} > threshold: {sim_validation_treshold}")
        
        return workingpoint_accurate

    def get_ycols_results(self, io_dict):
        """
        Extracts only the y_cols results from the io_dict.
        
        Args:
            io_dict (dict): Dictionary containing input and output values.
        
        Returns:
            dict: Dictionary with only y_cols results.
        """
        return {col: io_dict[col] for col in self.simulation_model.y_cols if col in io_dict}

    def get_xcols_results(self, io_dict):
        """
        Extracts only the y_cols results from the io_dict.
        
        Args:
            io_dict (dict): Dictionary containing input and output values.
        
        Returns:
            dict: Dictionary with only y_cols results.
        """
        return {col: io_dict[col] for col in self.simulation_model.x_cols if col in io_dict}


    def store_simulation_data(self, iodata, type):
        """
        Store simulation data for later use.
        
        Args:
            iodata (dict): Dictionary containing simulation input/output data.
            type (str): Type of data, e.g., 'validation' or 'direct_calibration'.
        """
        if self.simulation_data_collect is None:
            self.simulation_data_collect = {}

        self.simulation_data_collect.update({type:iodata})

    def get_simulation_data(self, type):
        """
        Get stored simulation data by type.
        
        Args:
            type (str): Type of data to retrieve, e.g., 'validation' or 'direct_calibration'.
        
        Returns:
            dict: Stored simulation data for the specified type.
        """

        iores = self.simulation_data_collect.get(type, None)
        x_data = {col: iores[col] for col in self.simulation_model.x_cols if col in iores}
        y_data = {col: iores[col] for col in self.simulation_model.y_cols if col in iores}

        return x_data, y_data
    
    def clear_simulation_data(self):
        """
        Clear all stored simulation data.
        """
        self.simulation_data_collect = None

    def _init_random_seed(self):
        seed = self.workflow_settings.get('workflow_random_seed', None)
        if seed is not None:
            np.random.seed(seed)
            self.logger.debug(f"Workflow random seed set to: {seed}")

    def _validate_models(self):
        if self.simulation_model is None:
            raise ValueError("Simulation model is not set.")
        if self.surrogate_model is None and self.calibration_type != CalibrationType.DIRECT:
            raise ValueError("Surrogate model is required unless calibration type is DIRECT.")
        if self.online_learner is None and self.calibration_type == CalibrationType.ONLINE:
            raise ValueError("Online learner is required for ONLINE calibration type.")

    def _init_calibration_settings(self):
        self.y_fitt = self.calibration_settings.get('y_fitt') or {
            param: {'Weight': 1, 'SDEV': 1} for param in self.surrogate_model.y_cols
        }
        self.fitparamlimit = self.calibration_settings.get('fitparamlimit')
        if self.fitparamlimit is None:
            raise ValueError("Fit parameter limits are not set.")

        self.particle_num = self.calibration_settings.get('particle_num', 20)
        if self.particle_num <= 0:
            raise ValueError("Particle number must be > 0.")

        self.particles_for_fittparams = {
            param: np.random.choice(
                np.arange(lim['min'], lim['max'], 1.0e-05), self.particle_num
            )
            for param, lim in self.fitparamlimit.items()
        }

        self.optim_iterations = self.calibration_settings.get('optim_iterations', 100)
        self.optim_c1 = self.calibration_settings.get('c1', 0.1)
        self.optim_c2 = self.calibration_settings.get('c2', 0.4)
        self.optim_w = self.calibration_settings.get('w', 0.7)
        self.optim_stopping_treshold = self.calibration_settings.get('stopping_treshold', 1e-6)
        self.optim_stopping_obj = self.calibration_settings.get('stopping_obj', 1e-4)
        self.optim_debug = self.calibration_settings.get('optim_debug', False)

    def _should_stop(self, i):
        stop_workingpoint = self.workflow_settings.get('stop_workingpoint', None)
        if stop_workingpoint is not None and i >= stop_workingpoint:
            self.logger.info(f"Stopping calibration at working point {stop_workingpoint}.")
            return True
        return False

    def _validate_calibration(self, pso, base_metrics):
        """
        Run simulation validation for the best particle (if not DIRECT),
        store validation IO, compute validation cost and check accuracy.
        Returns:
            sim_valid_result (dict) -- full simulation result (or default incaseofnooutput)
            sim_validation_cost (float|None)
            workingpoint_sim_valid_accurate (bool|None)
            used_surrogate_version (str|None)
        """
        debug_workflow = self.workflow_settings.get('debug_workflow', False)

        # default simulation result in case simulation fails
        sim_valid_result = self.simulation_model.incaseofnooutput if self.simulation_model else {}
        used_surrogate_version = None

        # what surrogate version was used (if any)
        if self.calibration_type != CalibrationType.DIRECT:
            if self.calibration_type == CalibrationType.SURROGATE:
                used_surrogate_version = self.surrogate_model.current_version if self.surrogate_model else None
            elif self.calibration_type == CalibrationType.ONLINE:
                used_surrogate_version = self.online_learner.surrogate.current_version if self.online_learner else None

        # run simulation validation only for surrogate/online base calibrations
        if self.calibration_type != CalibrationType.DIRECT:
            if debug_workflow:
                self.logger.info(f"Running simulation validation for the best particle: {base_metrics.get('gbest')}")

            if self.simulation_model is not None:
                try:
                    simresc = self.simulation_model.predict(base_metrics.get('gbest', {}))
                    if simresc:
                        sim_valid_result = simresc
                    else:
                        # keep default, attempt a restart as originally implemented
                        self.logger.error("Simulation validation failed: simulation returned None.")
                        try:
                            self.simulation_model.close()
                        except Exception:
                            self.logger.debug("Closing simulation model raised an exception (ignored).")
                        time.sleep(10)
                        self.logger.info("Restarting simulation model...")
                        self.simulation_model.load_model()
                except Exception as e:
                    self.logger.exception("Exception during simulation validation: %s", e)
                    # keep sim_valid_result as default

            # store validation data for later (used by online learning etc.)
            try:
                self.store_simulation_data({**(base_metrics.get('gbest') or {}), **(sim_valid_result or {})}, 'validation')
            except Exception:
                # fail-safe: don't break workflow if store fails
                self.logger.debug("Storing simulation validation data failed (ignored).")

        # pick y-results used to compute the same cost function as used in PSO
        if self.calibration_type == CalibrationType.DIRECT:
            sim_validation_yresults = self.get_ycols_results(base_metrics.get('best_io', {}))
        else:
            sim_validation_yresults = sim_valid_result

        sim_validation_cost = None
        workingpoint_sim_valid_accurate = None
        if sim_validation_yresults is not None:
            sim_validation_treshold = self.workflow_settings.get('sim_validation_treshold', self.optim_stopping_obj)
            try:
                sim_validation_cost = pso.calc_cost(sim_validation_yresults)
                workingpoint_sim_valid_accurate = self.check_accuracy(
                    sim_validation_cost=sim_validation_cost,
                    sim_validation_treshold=sim_validation_treshold,
                    debug_workflow=debug_workflow
                )
            except Exception as e:
                self.logger.exception("Error calculating validation cost: %s", e)
                sim_validation_cost = None
                workingpoint_sim_valid_accurate = None
        else:
            # Mirror original behaviour: signal that validation results are absent
            if debug_workflow:
                self.logger.error("Simulation validation results are None. Cannot proceed with accuracy check.")

        return sim_valid_result, sim_validation_cost, workingpoint_sim_valid_accurate, used_surrogate_version


    def _needs_direct_calibration(self, workingpoint_sim_valid_accurate):
        """
        Decide if a direct (simulation) calibration should be run
        when the base (surrogate or online) calibration is not accurate.
        """
        return (
            self.calibration_type != CalibrationType.DIRECT
            and self.run_direct_calib_for_not_accurate
            and not bool(workingpoint_sim_valid_accurate)
        )


    def _run_online_learning(self, sim_valid_result, workingpoint_sim_valid_accurate, direct_metrics=None):
        """
        Run online learning selection & addition of the selected simulation point(s).
        Returns a dict summarizing the online-learning step:
            {
            'selectiontype': 'validation'|'direct_calibration'|None,
            'new_data_selected': {...}|None,
            'elapsed': float|None,
            'selected_replay_indices': ...|None,
            'point_selection_type': original setting
            }
        """
        point_selection_type = self.workflow_settings.get('point_selection_type', None)
        selectiontype = None
        new_data_selected = None
        selected_replay_indices = None
        elapsed = None

        self.log_minor_milestone(f"[ONLINELEARNING] Running online learning with sample selection type: {point_selection_type}.")
        o_start = time.time()

        if self.online_learner is None:
            raise ValueError("Online learner is not initialized. It is mandatory to initialize online learner for the calibration workflow.")

        if sim_valid_result is None:
            self.logger.error("Simulation validation result is None. Cannot proceed with online learning.")
            return {'selectiontype': None, 'new_data_selected': None, 'elapsed': None, 'selected_replay_indices': None, 'point_selection_type': point_selection_type}

        # always-one: If direct calibration data exists and is accurate, use that; else use validation data if accurate.
        # always-both: Use both validation and direct calibration data if available and accurate.
        # conditional: Only use direct calibration data if it exists and is accurate.
        if point_selection_type == 'always-one':
            if direct_metrics:
                selectiontype = 'direct_calibration'
            elif workingpoint_sim_valid_accurate:
                selectiontype = 'validation'
            else:
                selectiontype = None
                self.logger.warning("No accurate working point found. Skipping online learning step.")
            self.logger.info(f"Selected point for online learning: {selectiontype}")
        elif point_selection_type in ('always-both', 'both'):
            selectiontype = []
            selectiontype.append('validation')
            if direct_metrics:
                selectiontype.append('direct_calibration')
            self.logger.info(f"Selected point for online learning: {selectiontype}")
        elif point_selection_type == 'conditional':
            if direct_metrics:
                selectiontype = 'direct_calibration'
            else:
                selectiontype = None
                self.logger.warning("No simulation working point found. Skipping online learning step.")
        else:
            selectiontype = None
            self.logger.error(f"Unknown point selection type: {point_selection_type}. Skipping online learning step.")

        # --- Data selection and addition ---
        x_op_sim = y_true_sim = None
        if selectiontype is not None:
            if isinstance(selectiontype, list):
                new_data_selected = {}
                for sel in selectiontype:
                    x_op, y_true = self.get_simulation_data(sel)
                    if x_op is not None and y_true is not None:
                        new_data_selected[sel] = {**(x_op or {}), **(y_true or {})}
                        # Add to online learner, but do NOT retrain here
                        self.online_learner.add_data_point(x_op, y_true)
                self.clear_simulation_data()
                self.online_learner.init_retraining()
                selected_replay_indices = getattr(self.online_learner, 'last_replay_indices', None)
                self.logger.info(f"New data points ({selectiontype}) added for online learning.")
            else:
                x_op_sim, y_true_sim = self.get_simulation_data(selectiontype)
                new_data_selected = {**(x_op_sim or {}), **(y_true_sim or {})}
                if x_op_sim is not None and y_true_sim is not None:
                    self.online_learner.add_data_point(x_op_sim, y_true_sim)
                    self.clear_simulation_data()
                    self.online_learner.init_retraining()
                    selected_replay_indices = getattr(self.online_learner, 'last_replay_indices', None)
                    self.logger.info(f"New data point ({selectiontype}) added for online learning.")
        else:
            self.logger.warning("No simulation data available for online learning. Skipping online learning step.")

        o_end = time.time()
        elapsed = o_end - o_start
        # This is for test purposes, in production it should be calculated different way as the calibration factors are not known calibration factors
        # In poroduction the calibration and validation process should be run with the new model and the old model to calculate the local improvement
        if self.next_data_item is not None:
            local_improvement = self.online_learner.calculate_local_improvement_for_next_point(self.next_data_item[self.online_learner.surrogate.x_cols].to_dict(), 
                                                                                               self.next_data_item[self.online_learner.surrogate.y_cols].to_dict())
        else: 
            local_improvement  = None 
        global_accuracy_retention = self.online_learner.calculate_global_accuracy_retention()

        return {
            'selectiontype': selectiontype,
            'new_data_selected': new_data_selected,
            'elapsed': elapsed,
            'selected_replay_indices': selected_replay_indices,
            'point_selection_type': point_selection_type,
            'local_improvement': local_improvement,
            'global_accuracy_retention': global_accuracy_retention,
        }


    def _compile_results(self, k, w_start_time, original_factors, y_true,
                        base_metrics, sim_validation_result, sim_validation_cost, workingpoint_sim_valid_accurate,
                        direct_metrics, online_metrics, used_surrogate_version):
        """
        Compose the final result dict for a single sample (mirrors original `rest` structure).
        """
        w_end_time = time.time()
        
        # surrogate version after the step
        if self.calibration_type == CalibrationType.SURROGATE:
            new_surrogate_version = self.surrogate_model.current_version if self.surrogate_model else None
        elif self.calibration_type == CalibrationType.ONLINE:
            new_surrogate_version = self.online_learner.surrogate.current_version if self.online_learner else None
        else:
            new_surrogate_version = None

        # online learning summary
        selected_replay_indices = online_metrics.get('selected_replay_indices') if online_metrics else None
        point_selection_type = online_metrics.get('point_selection_type') if online_metrics else self.workflow_settings.get('point_selection_type', None)
        new_data_selected = online_metrics.get('new_data_selected') if online_metrics else None
        onlline_learning_ellapsed_time = online_metrics.get('elapsed') if online_metrics else None
        local_improvement = online_metrics.get('local_improvement') if online_metrics else None
        global_accuracy_retention = online_metrics.get('global_accuracy_retention') if online_metrics else None

        # base calibration metrics
        costs_values = base_metrics.get('costs', []) if base_metrics else []
        bestio = base_metrics.get('best_io') if base_metrics else None
        gbest = base_metrics.get('gbest') if base_metrics else None
        gbest_obj = base_metrics.get('gbest_obj') if base_metrics else None
        base_elapsed = base_metrics.get('elapsed') if base_metrics else None

        # direct calibration metrics (may be None)
        if direct_metrics:
            d_elapsed_time = direct_metrics.get('elapsed')
            d_costs_values = direct_metrics.get('costs')
            d_gbest_obj = direct_metrics.get('gbest_obj')
            d_bestio = direct_metrics.get('best_io')
            d_sim_validation_cost = direct_metrics.get('sim_cost')
            d_workingpoint_sim_valid_accurate = direct_metrics.get('accurate')
        else:
            d_elapsed_time = d_costs_values = d_gbest_obj = d_bestio = d_sim_validation_cost = d_workingpoint_sim_valid_accurate = None


        if gbest is not None and d_bestio is None:
            final_calib_type = 'validation'
            final_io = {**self.get_ycols_results(bestio), **sim_validation_result}
            final_cost = sim_validation_cost
            final_accurate = workingpoint_sim_valid_accurate
        elif gbest is not None and d_bestio is not None:
            final_calib_type = 'direct'
            final_io = d_bestio
            final_cost = d_sim_validation_cost
            final_accurate = d_workingpoint_sim_valid_accurate
        else:
            final_calib_type = None
            final_io = None
            final_cost = None
            final_accurate = None

        rest = {
            'SampleID': k,
            'Total_workflow_EllapsedTime': w_end_time - w_start_time,
            'Calibration_type': self.calibration_type.value,
            'OriginalFactors': original_factors,
            'Target': y_true,

            'Calibration_Base_Type': CalibrationType.DIRECT if self.calibration_type == CalibrationType.DIRECT else CalibrationType.SURROGATE,
            'Calibration_Base_EllapsedTime': base_elapsed,
            'Calibration_Base_IterationNum': len(costs_values) if costs_values is not None else None,
            'Calibration_Base_Final_obj': gbest_obj,
            'Calibration_Base_FinalIO': bestio,
            'Calibration_Base_SurrogatePred': gbest if self.calibration_type != CalibrationType.DIRECT else None,
            'Validation_Base_SimulationRes': sim_validation_result,
            'Validation_Base_SimCost': sim_validation_cost,
            'Validation_Base_Accurate': workingpoint_sim_valid_accurate,
            'Validation_Base_Decision': None,

            'Calibration_Direct_EllapsedTime': d_elapsed_time,
            'Calibration_Direct_IterationNum': len(d_costs_values) if d_costs_values is not None else None,
            'Calibration_Direct_Final_obj': d_gbest_obj,
            'Calibration_Direct_FinalIO': d_bestio,
            'Calibration_Direct_SimCost': d_sim_validation_cost,
            'Calibration_Direct_Accuracy': d_workingpoint_sim_valid_accurate,
            'Calibration_Direct_Decision': None,

            'OnlineLearning_New_Data': new_data_selected,
            'OnlineLearning_New_SelectionType': point_selection_type,
            'OnlineLearning_New_DataType': online_metrics.get('selectiontype') if online_metrics else None,
            'OnlineLearning_EllapsedTime': onlline_learning_ellapsed_time,
            'OnlineLearning_SelectedReplayids': selected_replay_indices,
            'OnlineLearning_SelectedReplayIOs': None,
            'LocalImprovement': local_improvement,
            'GlobalRetention': global_accuracy_retention,
            'Used_Surrogate_version': used_surrogate_version,
            'New_Surrogate_version': new_surrogate_version,

            'Final_Calibration_Result_IOs': final_io,
            'Final_Calibration_SimCost': final_cost,
            'Final_Calibration_Accurate': final_accurate,
            'Final_Calibration_Type': final_calib_type,
        }

        return rest

    def _process_single_sample(self, k, item):
        w_start_time = time.time()

        particles = self._create_particles(item)
        y_true, original_factors = self._extract_targets_and_factors(item)

        # Base calibration
        pso, base_metrics = self._run_base_calibration(particles, y_true)

        # Validation
        sim_validation_results, sim_cost, is_accurate, used_surrogate_version = \
            self._validate_calibration(pso, base_metrics)

        # Optional direct calibration
        direct_metrics = None
        if self._needs_direct_calibration(is_accurate):
            new_particles = self._create_particles(item) # Reset particle do not use the base calibration end particles
            self.logger.debug(f"Direct calibration particles reinited {new_particles}")
            direct_metrics = self._run_direct_calibration(new_particles, y_true)

        # Online learning
        online_metrics = None
        if self.calibration_type == CalibrationType.ONLINE:
            online_metrics = self._run_online_learning(sim_validation_results, is_accurate, direct_metrics)

        # Compile results
        result_entry = self._compile_results(
            k, w_start_time, original_factors, y_true,
            base_metrics, sim_validation_results, sim_cost, is_accurate,
            direct_metrics, online_metrics, used_surrogate_version
        )
        return result_entry

    def _create_particles(self, item):
        return self.create_particles_with_full_state(
            item, self.simulation_model.x_cols, self.particles_for_fittparams
        )

    def _extract_targets_and_factors(self, item):
        y_true = {param: item[param] for param in self.surrogate_model.y_cols}
        original_factors = {param: item[param] for param in self.fitparamlimit.keys()}
        return y_true, original_factors

    def _init_pso(self, particles, y_true, init_type='base'):
        if init_type == 'base':
            base_calib_error_factor = self.workflow_settings.get('base_calibration_error_factor', 1)
        else:
            base_calib_error_factor = 1
            self.logger.debug(f"Init PSO base_calib_error_factor is set to {base_calib_error_factor}")
        return PSO.pso(
            particles=particles,
            opt_params=self.fitparamlimit.keys(),
            y_true=y_true,
            y_scaler=self.surrogate_model.scaler.y_scaler,
            iterations=self.optim_iterations,
            c1=self.optim_c1,
            c2=self.optim_c2,
            w=self.optim_w,
            stopping_treshold=self.optim_stopping_treshold*base_calib_error_factor,
            stopping_MSE=self.optim_stopping_obj*base_calib_error_factor,
            debug=self.optim_debug,
            y_fitt=self.y_fitt,
            fitparamlimit=self.fitparamlimit,
            logger=self.logger
        )

    def _run_direct_calibration(self, particles, y_true):
        """
        Run a direct (simulation-based) calibration when surrogate/online calibration
        results are not accurate enough.
        Returns a dict:
            {
                'elapsed': float,
                'costs': [...],
                'best_io': {...},
                'gbest_obj': float,
                'sim_cost': float,
                'accurate': bool
            }
        """
        debug_workflow = self.workflow_settings.get('debug_workflow', False)
        self.log_minor_milestone("[DIRECT-CALIBRATION] Running direct calibration as working point is not accurate enough.")
        if debug_workflow:
            self.logger.info("Running direct calibration with simulation model.")

        # Re-init PSO for direct calibration
        d_pso = self._init_pso(particles, y_true, init_type='direct')

        # Run PSO against the simulation model
        d_start_time = time.time()
        d_costs_values, d_iovalues, d_gbest, d_gbest_obj = d_pso.run_pso(self.simulation_model)
        d_end_time = time.time()
        d_elapsed_time = d_end_time - d_start_time

        # Get best IO (input + output) for direct calibration
        d_bestio = d_pso.get_xydicforbest(d_costs_values, d_iovalues, d_gbest)

        # Store simulation direct-calibration data for possible online-learning step
        try:
            self.store_simulation_data(d_bestio, 'direct_calibration')
        except Exception:
            self.logger.debug("Storing direct calibration data failed (ignored).")

        # Evaluate accuracy
        sim_validation_treshold = self.workflow_settings.get('sim_validation_treshold', self.optim_stopping_obj)
        d_sim_validation_cost = None
        d_workingpoint_sim_valid_accurate = None

        if d_bestio is not None:
            sim_direct_validation_yresults = self.get_ycols_results(d_bestio)
            if debug_workflow:
                self.logger.debug(f"Direct calibration validation results: {sim_direct_validation_yresults}")
            if sim_direct_validation_yresults is not None:
                d_sim_validation_cost = d_pso.calc_cost(sim_direct_validation_yresults)
                d_workingpoint_sim_valid_accurate = self.check_accuracy(
                    sim_validation_cost=d_sim_validation_cost,
                    sim_validation_treshold=sim_validation_treshold,
                    debug_workflow=debug_workflow
                )

        return {
            'elapsed': d_elapsed_time,
            'costs': d_costs_values,
            'best_io': d_bestio,
            'gbest_obj': d_gbest_obj,
            'sim_cost': d_sim_validation_cost,
            'accurate': d_workingpoint_sim_valid_accurate
        }

    def _run_base_calibration(self, particles, y_true):

        if self.calibration_type != CalibrationType.DIRECT:
            init_type = 'base'
        else:
            init_type = 'direct'
    
        pso = self._init_pso(particles, y_true, init_type=init_type)

        b_start_time = time.time()

        if self.calibration_type == CalibrationType.DIRECT:
            costs, ios, gbest, gbest_obj = pso.run_pso(self.simulation_model)
        elif self.calibration_type == CalibrationType.SURROGATE:
            costs, ios, gbest, gbest_obj = pso.run_pso(self.surrogate_model)
        elif self.calibration_type == CalibrationType.ONLINE:
            costs, ios, gbest, gbest_obj = pso.run_pso(self.online_learner.surrogate)
        else:
            raise ValueError(f"Unsupported calibration type: {self.calibration_type}")
        
        b_end_time = time.time()
        b_elapsed_time = b_end_time - b_start_time

        best_io = pso.get_xydicforbest(costs, ios, gbest)

        return pso, {
            'costs': costs,
            'best_io': best_io,
            'gbest_obj': gbest_obj,
            'gbest': gbest,
            'elapsed': b_elapsed_time
        }

    def run_calibration(self):
        """Main entry point for running the full calibration workflow."""
        self._init_random_seed()
        self._validate_models()
        self._init_calibration_settings()

        self.log_milestone("Starting calibration workflow.")
        total_result = []

        for i, (k, item) in enumerate(self.process_data.iterrows(), start=1):
            if self._should_stop(i):
                break
            self.next_data_item = self.process_data.iloc[i] if i < len(self.process_data) else None  # set next data item for online learning validation purpose

            self.log_minor_milestone(f"[PRIMARY-CALIBRATION] Processing sample {k} of {len(self.process_data)-1}.")
            result_entry = self._process_single_sample(k, item)
            total_result.append(result_entry)

        self.calibration_results = total_result
        self.logger.info("Calibration workflow completed.")

    def get_results(self):
        if self.calibration_results is None:
            raise ValueError("Calibration has not been run yet.")
        return self.calibration_results
    
    def save_results_df(self, filename):
        """
        Save the calibration results to a file. 
        Args:
            filename (str): The name of the file to save the results to.
        """
        if self.calibration_results is None:
            raise ValueError("Calibration has not been run yet. No results to save.")
        df = pd.DataFrame(self.calibration_results)
        df.to_excel(filename, index=False)
        self.logger.info(f"Calibration results saved to {filename}")