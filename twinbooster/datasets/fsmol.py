import json
import os
import pandas as pd


class FSMOLDataset:
    """
    Class for loading the FSMOL dataset. Provides methods for loading tasks from the train, test, and valid splits.
    """
    def __init__(self, dataset_dir: str = "fs-mol"):
        """
        :param dataset_dir: Location of the FSMOL dataset.
        """
        self.dataset_dir = dataset_dir
        self.split_mapping = self._load_split_mapping()
        self.max_task_index = self._get_max_task_index()
        self.lookup = self._load_lookup()

    def _load_split_mapping(self):
        mapping_file = os.path.join(self.dataset_dir, "fsmol-0.1.json")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError("fsmol.json file not found.")

        with open(mapping_file, "r") as file:
            split_mapping = json.load(file)

        new_split_mapping = split_mapping.copy()
        for split in split_mapping.keys():
            split_dir = os.path.join(self.dataset_dir, split)
            task_ids = [f.split(".")[0] for f in os.listdir(split_dir) if f.endswith(".jsonl.gz")]
            if len(task_ids) > len(split_mapping[split]):
                extra_ids = [task_id for task_id in task_ids if task_id not in split_mapping[split]]
                new_split_mapping[f"{split}_extra"] = extra_ids

        return new_split_mapping

    def _get_max_task_index(self):
        max_indexes = {}
        for split in self.split_mapping.keys():
            max_indexes[split] = len(self.split_mapping[split]) - 1

        return max_indexes

    def _load_lookup(self):
        lookup = pd.read_csv(os.path.join(self.dataset_dir, "fsmol_lookup.csv"))
        return lookup

    def load_task(self, split: str, task_index: int, return_aid: bool = False):
        """
        Load a task from the dataset.
        :param return_aid: If True it returns the AID of the task.
        :param split: Split to load the task from. Must be one of "train", "test", or "valid" (or "train_extra").
        :param task_index: Index of the task to load.
        :return: Pandas DataFrame containing the task. Columns are "SMILES", "Property", etc.
        """
        if split not in self.split_mapping.keys():
            raise ValueError("Split must be valid.")

        split_dir = os.path.join(self.dataset_dir, "train" if split == "train_extra" else split)

        if -1 * self.max_task_index[split] > task_index > self.max_task_index[split]:
            raise ValueError("Invalid task index.")

        task_file = self.split_mapping[split][task_index]
        task_path = os.path.join(split_dir, task_file + ".jsonl.gz")

        if return_aid:
            try:
                aid = self.lookup[self.lookup["chembl_id"] == task_file]["aid"].values[0]
                return pd.read_json(task_path, lines=True), aid

            except IndexError:
                Warning(f"Could not find AID for {task_file}.")
                return pd.read_json(task_path, lines=True), None

        else:
            return pd.read_json(task_path, lines=True)

    def load_train_task(self, task_index, return_aid):
        return self.load_task("train", task_index, return_aid)

    def load_test_task(self, task_index, return_aid):
        return self.load_task("test", task_index, return_aid)

    def load_valid_task(self, task_index, return_aid):
        return self.load_task("valid", task_index, return_aid)
