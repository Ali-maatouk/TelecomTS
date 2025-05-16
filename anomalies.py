import numpy as np
import matplotlib.pyplot as plt

class Anomaly:
    """
    Class to represent a network anomaly with multiple affected metrics.
    
    Parameters:
    - name: The name of the anomaly
    - metric_attributes: Dictionary of affected metrics, where:
        {
            "metric_name": (function_type, (param1_left, param1_right), (param2_left, param2_right), ...)
        }
      param_x is drawn uniformly between param_x_left and param_x_right
    """

    def __init__(self, name, metric_attributes):
        self.name = name
        self.metric_attributes = metric_attributes
        self.metric_instance_dict = {}

    def get_instance(self):
        """
        Creates a new instance of the anomaly by sampling parameters 
        for each metric and storing Metric class instances.
        """
        self.metric_instance_dict = {}

        for metric, (function_type, *param_ranges) in self.metric_attributes.items(): #*param_ranges collects tuples into list
            sampled_params = [np.random.uniform(low, high) for low, high in param_ranges]
            self.metric_instance_dict[metric] = Metric((metric, function_type, *sampled_params))#*sampled_params unpacks list

    def format_output(self):
        formatted_strings = [
            metric_instance.format_output() 
            for metric_instance in self.metric_instance_dict.values()
        ]
        return " ".join(formatted_strings)

    def apply_transformation(self, wireless_dataframe, occurrence_time, resolution_time):
        mask = (wireless_dataframe['date'] >= occurrence_time) & (wireless_dataframe['date'] <= resolution_time) 
        wireless_dataframe.loc[mask, "time_elapsed"] = (wireless_dataframe.loc[mask, "date"] - occurrence_time).dt.total_seconds()
        for metric_instance in self.metric_instance_dict.values():
            metric_instance.apply_transformation(wireless_dataframe, mask)

            if metric_instance.metric_name in {"UL_MCS", "DL_MCS", "TX_Bytes", "RX_Bytes"}:
                wireless_dataframe.loc[mask, metric_instance.metric_name] = wireless_dataframe.loc[mask, metric_instance.metric_name].round().astype(int)
        wireless_dataframe.drop(columns=["time_elapsed"], inplace=True, errors='ignore')


class Metric:
    """
    Class to model different types of metric transformations.
    
    Parameters:
    - metric_tuple: (metric_name, function_type, param1, param2, ...)
    """

    METRIC_NAME_MAP = {
        "UL_NumberOfPackets": "Uplink packet count",
        "DL_NumberOfPackets": "Downlink packet count",
        "UL_BLER": "Uplink block error rate",
        "DL_BLER": "Downlink block error rate",
        "RSRP": "Reference Signal Received Power",
        "UL_SNR": "Uplink Signal-to-Noise Ratio",
        "DL_SNR": "Downlink Signal-to-Noise Ratio",
        "Estimated_UL_Buffer": "Estimated uplink buffer backlog",
        "TX_Bytes": "Transmitted bytes",
        "RX_Bytes": "Received bytes",
        "PRB_Utilization_DL": "Downlink PRB utilization",
        "PRB_Utilization_UL": "Uplink PRB utilization",
        "PRBs_DL_Current": "Current Downlink PRBs",
        "PRBs_UL_Current": "Current Uplink PRBs",
        "DL_MCS": "Downlink MCS",
        "UL_MCS": "Uplink MCS",
        "UL_NPRB": "Number of PRBs allocated"
    }

    BOUNDS_MAP = {
        "UL_NumberOfPackets": (0, 10000),  
        "DL_NumberOfPackets": (0, 10000),  
        "UL_BLER": (0, 1),  
        "DL_BLER": (0, 1),  
        "RSRP": (-135, -40),  
        "UL_SNR": (-5, 30), 
        "DL_SNR": (-5, 30), 
        "Estimated_UL_Buffer": (0, 200000),  
        "TX_Bytes": (0, 1e8),  
        "RX_Bytes": (0, 1e8),  
        "PRB_Utilization_DL": (0, 100),  
        "PRB_Utilization_UL": (0, 100),  
        "PRBs_DL_Current": (0, 100),  
        "PRBs_UL_Current": (0, 100),  
        "DL_MCS": (0, 28),  
        "UL_MCS": (0, 28),  
        "UL_NPRB": (0, 100)
    }

    BOUND_STD_DICT = {
        "UL_NumberOfPackets": 500,
        "DL_NumberOfPackets": 500,
        "Estimated_UL_Buffer": 4000,
        "TX_Bytes": 2e6,
        "RX_Bytes": 2e6,
        "RSRP": 2,
        "UL_SNR": 3,
        "DL_SNR": 3,
        "PRB_Utilization_DL": 5,
        "PRB_Utilization_UL": 5,
        "PRBs_DL_Current": 5,
        "PRBs_UL_Current": 5,
        "UL_BLER": 0.04,
        "DL_BLER": 0.04,
        "UL_NPRB": 4
    }

    def __init__(self, metric_tuple):
        self.metric_name = metric_tuple[0]
        self.function_type = metric_tuple[1]
        self.params = metric_tuple[2:]

    def get_hard_soft_bounds(self, metric_name, original_data):
        upper_soft_only = {"PRB_Utilization_DL", "PRB_Utilization_UL", "PRBs_DL_Current", "PRBs_UL_Current", "UL_NumberOfPackets", "DL_NumberOfPackets", "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "UL_NPRB"}
        both_soft_bounds = {"RSRP", "UL_SNR", "DL_SNR"}

        unidirectional_soft = {"PRB_Utilization_DL", "PRB_Utilization_UL", "PRBs_DL_Current", "PRBs_UL_Current", "UL_NPRB"}
        bidirectional_soft = {"UL_NumberOfPackets", "DL_NumberOfPackets", "Estimated_UL_Buffer", "TX_Bytes", "RX_Bytes", "RSRP", "UL_SNR", "DL_SNR"}

        base_min, base_max = Metric.BOUNDS_MAP.get(metric_name, (-np.inf, np.inf))
        std = Metric.BOUND_STD_DICT.get(metric_name, 0.05 * (base_max - base_min))

        noise1 = np.random.normal(loc=0.0, scale=std)
        noise2 = np.random.normal(loc=0.0, scale=std)
        if metric_name in unidirectional_soft:
            noise1 = -abs(noise1)
            noise2 = -abs(noise2)

        sorted_series = original_data[metric_name].dropna().sort_values(ascending = False)
        threshold = 20

        if len(sorted_series) >= threshold:
            orig_data_min = sorted_series.iloc[-threshold]  
            orig_data_max = sorted_series.iloc[threshold - 1]
        else:
            orig_data_min = sorted_series.min()
            orig_data_max = sorted_series.max()
        
        if metric_name in upper_soft_only:
            base_max =  base_max + noise1
        elif metric_name in both_soft_bounds:
            base_min = base_min + noise1
            base_max = base_max + noise2

        return min(base_min, orig_data_min), max(base_max, orig_data_max)

    def apply_transformation(self, wireless_dataframe, mask):
        column = self.metric_name
        min_bound, max_bound = self.get_hard_soft_bounds(column, wireless_dataframe)
        time_elapsed = wireless_dataframe.loc[mask, "time_elapsed"]

        if self.function_type == "c_add":
            wireless_dataframe.loc[mask, column] += self.params[0]
            noise = np.random.normal(loc=0, scale = 0.015 * abs(self.params[0]), size=wireless_dataframe.loc[mask, column].shape)
            wireless_dataframe.loc[mask, column] += noise

        elif self.function_type == "c_multiply":
            wireless_dataframe.loc[mask, column] *= self.params[0]

        elif self.function_type == "linear":
            wireless_dataframe.loc[mask, column] += self.params[0] * time_elapsed
            noise = np.random.normal(loc=0, scale = 1.5 * abs(self.params[0]), size=wireless_dataframe.loc[mask, column].shape) #noise for linear and sinusoidal addition
            wireless_dataframe.loc[mask, column] += noise

        elif self.function_type == "exp_growth":
            growth_rate = self.params[0]

            #apply kernel smoothing
            time_vals = wireless_dataframe.loc[mask, "time_elapsed"].to_numpy()
            original_values = wireless_dataframe.loc[mask, column].to_numpy()
            bandwidth = 2
            diffs = time_vals[:, np.newaxis] - time_vals[np.newaxis, :]
            weights = np.exp(-0.5 * (diffs / bandwidth) ** 2)
            weights /= weights.sum(axis=1, keepdims=True)

            smoothed_values = weights @ original_values

            noise_std = 0.005 * max_bound  
            if self.metric_name in ["UL_BLER", "DL_BLER", "DL_NumberOfPackets", "UL_NumberOfPackets", "UL_NPRB"]:
                noise_std = 0.001 * max_bound
            positive_noise = np.abs(np.random.normal(loc=0.0, scale=noise_std, size=original_values.shape)) + noise_std / 4
            if self.metric_name in ["Estimated_UL_Buffer", "PRB_Utilization_UL", "PRB_Utilization_DL"]: 
                positive_noise = 0
            deviation = original_values - smoothed_values
            wireless_dataframe.loc[mask, column] = (smoothed_values + positive_noise) * np.exp(growth_rate * wireless_dataframe.loc[mask, "time_elapsed"])
            wireless_dataframe.loc[mask, column] += deviation


        elif self.function_type == "logistic_growth":
            growth_rate = self.params[0]
            initial_value = wireless_dataframe.loc[mask, column]
            diff = max_bound - initial_value
            start_point = -2 * np.log((3 + np.sqrt(3))/(3 - np.sqrt(3))) / growth_rate

            #comes from solving for an "inflection point" of the sigmoid function
            wireless_dataframe.loc[mask, column] = initial_value + diff / (1 + np.exp(-growth_rate * (time_elapsed + start_point))) - diff / (1 + np.exp(-growth_rate * start_point))
            wireless_dataframe.loc[mask, column] += np.random.normal(loc=0, scale=0.005 * abs(diff), size=wireless_dataframe.loc[mask, column].shape)

        elif self.function_type == "log_decay":
            wireless_dataframe.loc[mask, column] -= self.params[0] * wireless_dataframe.loc[mask, column] * np.log1p(time_elapsed)

        elif self.function_type == "sinusoidal_add":
            amplitude = self.params[0]
            frequency = self.params[1]
            if len(self.params) > 2:
                offset = self.params[2]
                wireless_dataframe.loc[mask, column] += offset
            wireless_dataframe.loc[mask, column] += amplitude * np.sin(2 * np.pi * frequency * time_elapsed)
            noise = np.random.normal(loc=0, scale= abs(amplitude) / 20, size=wireless_dataframe.loc[mask, column].shape) #noise for linear and sinusoidal addition
            wireless_dataframe.loc[mask, column] += noise

        elif self.function_type == "sinusoidal_multiply":
            amplitude = self.params[0]
            if self.metric_name == "Resource Allocation Bugs":
                noise = np.random.normal(loc=0, scale= abs(amplitude) / 3, size=wireless_dataframe.loc[mask, column].shape)
                amplitude = noise + amplitude
            frequency = self.params[1]
            if len(self.params) > 2:
                offset = self.params[2]
                wireless_dataframe.loc[mask, column] *= offset
            wireless_dataframe.loc[mask, column] *= (1 + amplitude * np.sin(2 * np.pi * frequency * time_elapsed))

        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
        
        wireless_dataframe.loc[mask, column] = np.clip(wireless_dataframe.loc[mask, column], min_bound, max_bound)
        
        if column in ["Estimated_UL_Buffer", "UL_BLER", "DL_BLER", "UL_NumberOfPackets", "DL_NumberOfPackets", "RSRP", "UL_SNR", "DL_SNR", "PRB_Utilization_DL", "PRB_Utilization_UL", "PRBs_DL_Current", "PRBs_UL_Current", "UL_NPRB"]:
            threshold = 0.5 * Metric.BOUND_STD_DICT[column]
            near_max = (wireless_dataframe[column] - max_bound).abs() <= threshold
            near_max = near_max & mask
            noise_max = np.abs(np.random.normal(loc=0, scale=threshold, size=near_max.sum()))
            wireless_dataframe.loc[mask & near_max, column] -= noise_max
        
        if column in ["RSRP", "UL_SNR", "DL_SNR"]:
            threshold = 0.5 * Metric.BOUND_STD_DICT[column]
            near_min = (wireless_dataframe[column] - min_bound).abs() <= threshold
            near_min = near_min & mask
            noise_min = np.abs(np.random.normal(loc=0, scale=threshold, size=near_min.sum()))
            wireless_dataframe.loc[mask & near_min, column] += noise_min
        #if each item is close to the bound, add unidirectional gaussian noise

    
    def format_output(self):
        colloquial_name = self.METRIC_NAME_MAP.get(self.metric_name, self.metric_name)
        value = self.params[0]
        abs_value = abs(value) 

        if self.function_type == "c_add":
            return f"{colloquial_name} {'increased' if value > 0 else 'decreased'} by {abs_value:.2f}."

        elif self.function_type == "c_multiply":
            percent_change = (value - 1) * 100
            return f"{colloquial_name} {'increased' if value > 1 else 'decreased'} by {abs(percent_change):.2f}%."
        
        elif self.function_type == "linear":
            return f"{colloquial_name} {'increased' if value > 0 else 'decreased'} linearly."

        elif self.function_type == "exp_growth":
            return f"{colloquial_name} experienced exponential {'growth' if value > 0 else 'decay'}."

        elif self.function_type == "logistic_growth":
            return f"{colloquial_name} experienced logistic {'growth' if value > 0 else 'decay'}."

        elif self.function_type == "log_decay":
            return f"{colloquial_name} followed a logarithmic {'growth' if value > 0 else 'decay'} pattern."

        elif self.function_type == "sinusoidal_add":
            return f"{colloquial_name} fluctuated periodically with an amplitude of {self.params[0]:.2f} and a frequency of {self.params[1]:.4f} Hz."

        elif self.function_type == "sinusoidal_multiply":
            return f"{colloquial_name} oscillated multiplicatively with an amplitude factor of {self.params[0]:.2f} and a frequency of {self.params[1]:.4f} Hz."
        
        return f"{colloquial_name} remained unchanged."
    
    def apply_kernel_smoothing(self, column, time_vals, original_values, smoothed_values, old_exp, wireless_dataframe, mask, bandwidth):
        plt.figure(figsize=(12, 6))
        plt.plot(time_vals, original_values, 'b-', alpha=0.5, label='Original Values')
        plt.plot(time_vals, smoothed_values, 'r-', linewidth=2, label='Smoothed Values')
        plt.plot(time_vals, old_exp, 'y-', alpha=0.5, label='Old Exponential Values')
        plt.plot(time_vals, wireless_dataframe.loc[mask, column], 'g-', alpha=0.5, label='Smoothed + Noise Values')
            
        plt.xlabel('Time Elapsed (deciseconds)')
        plt.ylabel(column)
        ymin, ymax = Metric.BOUNDS_MAP[column]
        ymax = min(ymax, old_exp.max() * 1.1)
        plt.ylim(ymin, ymax)
        plt.title(f'Original vs Smoothed Values (Bandwidth = {bandwidth}s)\nColumn: {column}')
        plt.legend()
        plt.grid(True)
            
        # Save the plot
        filename = f"kernel_smoothing/smoothed_vs_original_{column}_{bandwidth}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        if column in ["UL_NumberOfPackets", "DL_NumberOfPackets"]:
            plt.figure(figsize=(12, 6))
            plt.plot(time_vals, original_values, 'b-', alpha=0.5, label='Original Values')
            plt.plot(time_vals, smoothed_values, 'r-', linewidth=2, label='Smoothed Values')
            plt.xlabel('Time Elapsed (deciseconds)')
            plt.ylabel(column)
            plt.title(f'Original vs Smoothed Values (Bandwidth = {bandwidth}s)\nColumn: {column}')
            plt.legend()
            plt.grid(True)
            filename = f"kernel_smoothing/non_exp_smoothed_vs_original_{column}_{bandwidth}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
