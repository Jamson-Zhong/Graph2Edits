import argparse
import os
import sys
from collections import Counter
from typing import Any, List

import joblib
import pandas as pd
from rdkit import Chem

from utils.generate_edits import generate_reaction_edits


def check_edits(edits: List):
    for edit in edits:
        if edit[0] == 'Add Bond':
            return False

    return True


def preprocessing(rxns: List, args: Any, rxn_classes: List = [], rxns_id=[]) -> None:
    """
    preprocess reactions data to get edits
    """
    rxns_data = []
    counter = []
    all_edits = {}

    savedir = f'data/{args.dataset}/{args.mode}'
    os.makedirs(savedir, exist_ok=True)

    for idx, rxn_smi in enumerate(rxns):
        r, p = rxn_smi.split('>>')
        prod_mol = Chem.MolFromSmiles(p)

        if (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
            print(
                f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        react_mol = Chem.MolFromSmiles(r)

        if (react_mol is None) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
            print(
                f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        try:
            if args.dataset == 'uspto_50k':
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=args.kekulize, rxn_class=int(
                    rxn_classes[idx]) - 1, rxn_id=rxns_id[idx])
            else:
                rxn_data = generate_reaction_edits(
                    rxn_smi, kekulize=args.kekulize)
        except:
            print(f'Failed to extract reaction data, skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        edits_accepted = check_edits(rxn_data.edits)
        if not edits_accepted:
            print(f'Edit: Add new bond. Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        # if args.dataset == 'uspto_full':
        #     if len(rxn_data.edits) > 9 or len(rxn_data.edits) == 1:
        #         print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')
        #         print()
        #         sys.stdout.flush()
        #         continue

        rxns_data.append(rxn_data)

        if (idx % args.print_every == 0) and idx:
            print(f'{idx}/{len(rxns)} {args.mode} reactions processed.')
            sys.stdout.flush()

    print(f'All {args.mode} reactions complete.')
    sys.stdout.flush()

    save_file = os.path.join(savedir, f'{args.mode}.file')
    if args.kekulize:
        save_file += '.kekulized'

    if args.mode == 'train':
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit not in all_edits:
                    all_edits[edit] = 1
                else:
                    all_edits[edit] += 1

        atom_edits = []
        bond_edits = []
        lg_edits = []
        atom_lg_edits = []

        if args.dataset == 'uspto_50k':
            for edit, num in all_edits.items():
                if edit[0] == 'Change Atom':
                    atom_edits.append(edit)
                    atom_lg_edits.append(edit)
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                    bond_edits.append(edit)
                elif edit[0] == 'Attaching LG':
                    lg_edits.append(edit)
            atom_lg_edits.extend(lg_edits)

        elif args.dataset == 'uspto_full':
            for edit, num in all_edits.items():
                if edit[0] == 'Change Atom':
                    atom_edits.append(edit)
                    atom_lg_edits.append(edit)
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                    bond_edits.append(edit)
                elif edit[0] == 'Attaching LG' and num >= 50:
                    lg_edits.append(edit)
            atom_lg_edits.extend(lg_edits)

        print(atom_edits)
        print(bond_edits)
        print(lg_edits)

        filter_rxns_data = []
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit[0] == 'Attaching LG' and edit not in lg_edits:
                    print(
                        f'The number of {edit} in training set is very small, skipping reaction')
                    rxn_data = None
            if rxn_data is not None:
                counter.append(len(rxn_data.edits))
                filter_rxns_data.append(rxn_data)

        print(Counter(counter))

        joblib.dump(filter_rxns_data, save_file, compress=3)
        joblib.dump(atom_edits, os.path.join(savedir, 'atom_vocab.txt'))
        joblib.dump(bond_edits, os.path.join(savedir, 'bond_vocab.txt'))
        joblib.dump(lg_edits, os.path.join(savedir, 'lg_vocab.txt'))
        joblib.dump(atom_lg_edits, os.path.join(savedir, 'atom_lg_vocab.txt'))
    else:
        bond_vocab_file = f'data/{args.dataset}/train/bond_vocab.txt'
        atom_vocab_file = f'data/{args.dataset}/train/atom_lg_vocab.txt'
        bond_vocab = joblib.load(bond_vocab_file)
        atom_vocab = joblib.load(atom_vocab_file)
        bond_vocab.extend(atom_vocab)
        all_edits = bond_vocab

        cover_num = 0
        for idx, rxn_data in enumerate(rxns_data):
            cover = True
            for edit in rxn_data.edits:
                if edit != 'Terminate' and edit not in all_edits:
                    print(f'{edit} in {args.mode} is not in train set')
                    cover = False
            if cover:
                cover_num += 1

            counter.append(len(rxn_data.edits))

        print(Counter(counter))
        print(f'The cover rate is {cover_num}/{len(rxns_data)}')
        joblib.dump(rxns_data, save_file, compress=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: USPTO_50k or uspto_full')
    parser.add_argument('--mode', type=str, default='train',
                        help='Type of dataset being prepared: train or valid or test')
    parser.add_argument('--print_every', type=int,
                        default=1000, help='Print during preprocessing')
    parser.add_argument('--kekulize', default=True, action='store_true',
                        help='Whether to kekulize mols during training')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    datadir = f'data/{args.dataset}/'
    rxn_key = 'reactants>reagents>production'
    if args.dataset == 'uspto_50k':
        filename = f'canonicalized_{args.mode}.csv'
        df = pd.read_csv(os.path.join(datadir, filename))
        preprocessing(rxns=df[rxn_key], args=args,
                      rxn_classes=df['class'], rxns_id=df['id'])
    else:
        filename = f'raw_{args.mode}.csv'
        df = pd.read_csv(os.path.join(datadir, filename))
        preprocessing(rxns=df[rxn_key], args=args)


if __name__ == '__main__':
    main()
