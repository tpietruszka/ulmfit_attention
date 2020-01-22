import os
from pathlib import Path
import traceback
import time
from sacred import observers, Experiment
from ulmfit_attention.task_queue import TaskQueue, Task
from ulmfit_attention import scenarios  # TODO: import from cwd, not relative to this source

# TODO: get this from command line or sth
db_name = 'ulmfit_attention'  # TODO: maybe set default to folder name?
mongo_uri = 'localhost:27017'
tasks_dir = Path(os.path.dirname(os.path.realpath(os.getcwd()))) / 'tasks'

sleep_time = 30


def run_from_queue():
    observer = observers.MongoObserver.create(mongo_uri, db_name=db_name)
    q = TaskQueue(mongo_uri, db_name, tasks_dir)
    while True:
        t = q.fetch_one()
        if t is None:
            print("No availale tasks in the queue. Sleeping.")
            time.sleep(sleep_time)
            continue
        try:
            run_task(t, observer)
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            q.remove(t)


def run_task(task: Task, observer: observers.RunObserver):
    ex = Experiment(task.name, interactive=True)
    ex.observers.append(observer)
    ex.add_config(task.params)
    ex.add_config(str(task.description_file))

    @ex.main
    def run_main(_config, _run):
        scenario = scenarios.Scenario.from_config(_config['scenario'])
        res = scenario.single_run(_config)
        _run.info = res[1]
        return res[0]

    run_res = ex.run()
    return run_res


if __name__ == "__main__":
    run_from_queue()
