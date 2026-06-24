import json
from collections import Counter

def analyze_results(confidences):
    print(Counter(list(confidences.values())))

def select_confidences(path2outputs, path2save):
    confidences = {}
    with open(path2outputs, 'r', encoding='utf-8') as file:
        prev_line = next(file)
        
        for line in file:
            if line == "<BEGIN RESPONSE>\n":
                pubid = int(prev_line)
            elif "confidence_score" in line:
                idx = line.find("confidence_score")
                if pubid not in confidences:
                    confidences[pubid] = []
                confidences[pubid].append(int(line[(idx + 19):]))
            prev_line = line

    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(confidences, f, indent=4)


def select_difficulties(path2outputs, path2save):
    difficulties = {}
    with open(path2outputs, 'r', encoding='utf-8') as file:
        prev_line = next(file)
        
        for line in file:
            if line == "<BEGIN RESPONSE>\n":
                pubid = int(prev_line)
            elif "difficulty" in line:
                idx = line.find("difficulty")
                if pubid not in difficulties:
                    difficulties[pubid] = []
                difficulties[pubid].append(int(line[(idx + 13):]))
            prev_line = line

    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(difficulties, f, indent=4)


def select_answers(path2outputs, path2save):
    answers = {}
    with open(path2outputs, 'r', encoding='utf-8') as file:
        prev_line = next(file)
        
        for line in file:
            if line == "<BEGIN RESPONSE>\n":
                pubid = int(prev_line)
            elif "\"answer\"" in line:
                idx = line.find("\"answer\"")
                if pubid not in answers:
                    answers[pubid] = []
                answers[pubid].append(str(line[(idx + 11):-3]))
            prev_line = line

    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4)


def select_meaningful_answers(path2outputs, path2save):
    # matching_dict = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.8, 6: 1.0}
    # matching_dict = {0: [1, 0, 0], 1: [0.75, 0.25, 0], 2: [0.5, 0.25, 0.25], 3: [0.33, 0.34, 0.33], 4: [0.25, 0.25, 0.5], 5: [0., 0.25, 0.75], 6: [0, 0, 1]}
    matching_dict = {0: [1, 0, 0], 1: [0.7, 0.2, 0.1], 2: [0.5, 0.25, 0.25], 3: [0.33, 0.34, 0.33], 4: [0.25, 0.25, 0.5], 5: [0.1, 0.2, 0.7], 6: [0, 0, 1],
                     7: [0, 1, 0]}
    answers = {}
    with open(path2outputs, 'r', encoding='utf-8') as file:
        prev_line = next(file)
        
        for line in file:
            if line == "<BEGIN RESPONSE>\n":
                pubid = int(prev_line)
            elif line == "<END RESPONSE>\n":
                if pubid not in answers:
                    answers[pubid] = []
                answers[pubid].append(matching_dict[int(prev_line[:-1])])
            # elif "\"answer\"" in line:
            #     idx = line.find("\"answer\"")
            #     if pubid not in answers:
            #         answers[pubid] = []
            #     answers[pubid].append(matching_dict[int(line[(idx + 10):-1])])
            prev_line = line

    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4)


def select_synonyms(path2outputs, path2save):
    answers = {}
    with open(path2outputs, 'r', encoding='utf-8') as file:
        prev_line = next(file)
        
        for line in file:
            if line == "<BEGIN RESPONSE>\n":
                pubid = int(prev_line)
            elif line == "<END RESPONSE>\n":
                if pubid not in answers:
                    answers[pubid] = []
                answers[pubid].append(prev_line[:-1])
            prev_line = line

    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4)


def main():
    data_dir = "."
    path2outputs = f"{data_dir}/mistral_synonyms.txt"
    select_synonyms(path2outputs, path2save=f"{data_dir}/synonyms.json")
    # select_confidences(path2outputs, path2save=f"{data_dir}/mistral-confidences_1-20.json")
    # select_answers(path2outputs, path2save=f"{data_dir}/mistral-answers_1-20.json")
    # select_difficulties(path2outputs, path2save=f"{data_dir}/mistral-difficulties.json")
    # select_meaningful_answers(path2outputs, path2save=f"{data_dir}/mistral-meaningful-answers.json")
    # analyze_results(confidences)


if __name__ == "__main__":
    main()
