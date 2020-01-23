from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import *
import datetime
from dataclasses import dataclass
from pathlib import Path

RunId = ObjectId


@dataclass
class QueuedRun:
    id: RunId
    task_name: str
    params: Dict
    task_description_file: Path


class RunQueue:
    """
    Representation of the task queue, returning only the tasks defined in the local tasks_dir
    """
    collection = 'queue'
    id_field = '_id'
    taskname_field = 'task_name'
    params_field = 'params'
    status_field = 'status'
    status_taken = 'TAKEN'
    time_inserted_field = 'time_inserted'
    time_taken_field = 'time_taken'

    def __init__(self, mongo_uri: str, db_name: str, tasks_dir: Union[str, Path]):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.tasks_dir = Path(tasks_dir)
        self.client = MongoClient(mongo_uri)
        self.queue = self.client[self.db_name][self.collection]

    def get_available_tasks(self) -> List[str]:
        return [f.stem for f in self.tasks_dir.iterdir() if f.is_file and f.suffix == '.json']

    def get_task_path(self, task_name: str) -> Path:
        return self.tasks_dir / f'{task_name}.json'

    def fetch_one(self) -> Optional[QueuedRun]:
        """Returns a task to compute, marking it as taken, but not fully removing from the queue"""
        available_tasks = self.get_available_tasks()
        query = {self.taskname_field: {'$in': available_tasks},
                 self.status_field: {'$ne': self.status_taken}}
        update = {'$set': {
            self.status_field: self.status_taken,
            self.time_taken_field: datetime.datetime.utcnow(),
        }}
        t = self.queue.find_one_and_update(query, update)
        if t is None:
            return None
        task = QueuedRun(id=t[self.id_field], task_name=t[self.taskname_field],
                         params=t[self.params_field], task_description_file=self.get_task_path(t[self.taskname_field]))
        return task

    def remove(self, task: QueuedRun) -> int:
        """Permanently removes the given task from the queue"""
        res = self.queue.delete_one({self.id_field: task.id})
        return res.deleted_count

    def submit(self, task_name: str, params: Dict) -> RunId:
        res = self.queue.insert_one({
            self.taskname_field: task_name,
            self.params_field: params,
            self.time_inserted_field: datetime.datetime.utcnow(),
        })
        return res.inserted_id
