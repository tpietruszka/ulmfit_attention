from pathlib import Path
import traceback
import time
import argparse
from sacred import observers, Experiment
from ulmfit_attention.run_queue import RunQueue, QueuedRun
from ulmfit_attention import scenarios  # TODO: import from cwd, not relative to this source


def process_queue(tasks_dir: Path, db_name: str, mongo_uri: str, sleep_time: int):
    observer = observers.MongoObserver.create(mongo_uri, db_name=db_name)
    q = RunQueue(mongo_uri, db_name, tasks_dir)
    while True:
        t = q.fetch_one()
        if t is None:
            print("No availale tasks in the queue. Sleeping.")
            time.sleep(sleep_time)
            continue
        try:
            single_run(t, observer)
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            q.remove(t)


def single_run(to_run: QueuedRun, observer: observers.RunObserver):
    ex = Experiment(to_run.task_name, interactive=True)
    ex.observers.append(observer)
    ex.add_config(to_run.params)
    ex.add_config(str(to_run.task_description_file))

    @ex.main
    def ex_main(_config, _run):
        scenario = scenarios.Scenario.from_config(_config['scenario'])
        res = scenario.single_run(_config)
        _run.info = res[1]
        return res[0]

    run_res = ex.run()
    return run_res


def main():
    desc = "Run experiments from a MongoDB-based queue. \nShould be ran from a folder containing " \
           "a `scenarios` module, defining the possible scenarios to run. Concrete tasks - " \
           "sets of parameters for scenarios - should be defined in json files, in the `tasks_dir` folder"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('tasks_dir', help='Path to a folder storing task json files, absolute or relative',
                      type=lambda s: Path(s).resolve())
    parser.add_argument('db_name', help='MongoDB database name')
    parser.add_argument('--mongo-uri', help='URI of the MongoDB server instance', default='localhost:27017')
    parser.add_argument('--sleep-time', type=int, default=30)
    args = parser.parse_args()
    process_queue(args.tasks_dir, args.db_name, args.mongo_uri, args.sleep_time)


if __name__ == "__main__":
    main()
