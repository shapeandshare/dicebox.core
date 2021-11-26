"""
This module contains code responsible for the review and selection
from prior population evolution outputs
"""
import json
from copy import deepcopy
from pathlib import Path

from shapeandshare.dicebox.config.dicebox_config import DiceboxConfig


def build_population_level_report():
    config_file = "./projects/mnist/dicebox.config"
    dicebox_config: DiceboxConfig = DiceboxConfig(config_file=config_file)
    population_directory: Path = Path(dicebox_config.POPULATION_DIR)
    for population in population_directory.glob("*"):
        generations = [name for name in population.glob("*") if name.is_dir()]
        print(f"population ({population.name}), generations: ({len(generations)})")
        fittest = []
        for generation_number in range(len(generations)):
            generation_path = population / str(generation_number)
            if generation_path in generations:
                generation_data_filename = (generation_path / "population.json").resolve().as_posix()
                with open(generation_data_filename, "r") as file:
                    generation_data = json.load(file)
                    average_accuracy = generation_data["average_accuracy"]
                    print(f"({str(generation_number)}) average accuracy: ({average_accuracy})")
                    for individual in generation_data["population"]:
                        if individual["accuracy"] >= average_accuracy:
                            print(f"individual accuracy (above population average): ({individual['accuracy']})")
                            if individual not in fittest:
                                fittest.append(individual)
            else:
                print(f"Unable to find generation ({str(generation_number)}) in population")

        top_performers = []
        max_top_performers = 3

        performer_scores = []
        for individual in fittest:
            if individual["accuracy"] not in performer_scores:
                performer_scores.append(individual["accuracy"])
        performer_scores.sort(reverse=True)
        del performer_scores[max_top_performers:]
        for score in performer_scores:
            for individual in fittest:
                if individual["accuracy"] == score:
                    top_performers.append(individual)

        with open((population / "fittest.json").resolve().as_posix(), "w") as file:
            file.write(json.dumps({"top_performers": top_performers, "above_pop_avg": fittest}, indent=4))
            # json.dump(file, fittest)
        # # for generation in generations:
        #     generation_data_filename = (generation / "population.json").resolve().as_posix()
        #     with open(generation_data_filename, "r") as file:
        #         generation_data = json.load(file)
        #         average_accuracy = generation_data["average_accuracy"]
        #         print(f"average accuracy: ({average_accuracy})")


def build_project_level_report():
    config_file = "./projects/mnist/dicebox.config"
    dicebox_config: DiceboxConfig = DiceboxConfig(config_file=config_file)
    population_directory: Path = Path(dicebox_config.POPULATION_DIR)

    populations = [name for name in population_directory.glob("*") if name.is_dir()]
    fittest = []
    for population in populations:
        print(f"population ({population.name})")
        with open((population / "fittest.json").resolve().as_posix(), "r") as file:
            population_fittest_report = json.load(file)
            population_top_performers = population_fittest_report["top_performers"]

            # limit per population moves
            for performer in population_top_performers[:3]:
                if performer not in fittest:
                    fittest.append(performer)

    top_performers = []
    max_top_performers = 3

    performer_scores = []
    for individual in fittest:
        if individual["accuracy"] not in performer_scores:
            performer_scores.append(individual["accuracy"])
    performer_scores.sort(reverse=True)
    del performer_scores[max_top_performers:]
    for score in performer_scores:
        for individual in fittest:
            if individual["accuracy"] == score:
                top_performers.append(individual)

    with open((population_directory / "fittest.json").resolve().as_posix(), "w") as file:
        file.write(json.dumps({"top_performers": top_performers, "fittest": fittest}, indent=4))


if __name__ == "__main__":
    # build_population_level_report()
    build_project_level_report()
