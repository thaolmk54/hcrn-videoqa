import os
import pandas as pd
import json
from datautils import utils
import nltk

import pickle
import numpy as np


def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    annotation = pd.read_csv(args.annotation_file.format(args.question_type), delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])

        if args.question_type == "frameqa":
            answer = answers[idx]
            if answer in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][answer]
            elif mode in ['train']:
                answer = 0
            elif mode in ['val', 'test']:
                answer = 1
        else:
            answer = max(int(answers[idx]), 1)
        all_answers.append(answer)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])
        # grounthtruth
        answer = int(answers[idx])
        correct_answers.append(answer)
        # answer candidates
        candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []
        for ans in candidates:
            ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
    for ans_cands in all_answer_cands_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_cand_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
    all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
    all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
    print(all_answer_cands_encoded.shape)

    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_openended(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])

    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}

        if args.question_type == "frameqa":
            for i, answer in enumerate(answers):
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

            answer_token_to_idx = {'<UNK>': 0}
            for token in answer_cnt:
                answer_token_to_idx[token] = len(answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
        elif args.question_type == 'count':
            answer_token_to_idx = {'<UNK>': 0}

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
        }

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]

        openeded_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
        openeded_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='test')




def process_questions_mulchoices(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for candidates in ans_candidates:
            for ans in candidates:
                ans = ans.lower()
                for token in nltk.word_tokenize(ans):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx,
        }

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]
        train_ans_candidates = ans_candidates[:split, :]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]
        val_ans_candidates = ans_candidates[split:, :]

        multichoice_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
        multichoice_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers,
                                  val_ans_candidates, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers,
                                  ans_candidates, mode='test')
