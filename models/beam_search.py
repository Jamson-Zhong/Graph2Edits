import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from rdkit import Chem

from utils.rxn_graphs import MolGraph
from utils.collate_fn import get_batch_graphs
from prepare_data import apply_edit_to_mol
from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)


class BeamSearch:
    def __init__(self, model, step_beam_size, beam_size, use_rxn_class):
        self.model = model
        self.step_beam_size = step_beam_size
        self.beam_size = beam_size
        self.use_rxn_class = use_rxn_class

    def process_path(self, path, rxn_class):
        new_paths = []

        prod_mol = path['prod_mol']
        steps = path['steps'] + 1
        prod_tensors = self.model.to_device(path['tensors'])
        edit_logits, state, state_scope = self.model.compute_edit_scores(
            prod_tensors, path['scopes'], path['state'], path['state_scope'])
        edit_logits = edit_logits[0]
        edit_logits = F.softmax(edit_logits, dim=-1)

        k = self.step_beam_size
        top_k_vals, top_k_idxs = torch.topk(edit_logits, k=k)

        for beam_idx, (topk_idx, val) in enumerate(zip(*(top_k_idxs, top_k_vals))):
            edit, edit_atom = self.get_edit_from_logits(
                mol=prod_mol, edit_logits=edit_logits, idx=topk_idx, val=val)
            val = round(val.item(), 4)
            new_prob = path['prob'] * val

            if edit == 'Terminate':
                edits_prob, edits = [], []
                edits_prob.extend(path['edits_prob'])
                edits_prob.append(val)
                edits.extend(path['edits'])
                edits.append(edit)
                final_path = {
                    'prod_mol': prod_mol,
                    'steps': steps,
                    'prob': new_prob,
                    'edits_prob': edits_prob,
                    'tensors': path['tensors'],
                    'scopes': path['scopes'],
                    'state': state,
                    'state_scope': state_scope,
                    'edits': edits,
                    'edits_atom': path['edits_atom'],
                    'finished': True,
                }
                new_paths.append(final_path)

            else:
                try:
                    int_mol = apply_edit_to_mol(mol=Chem.Mol(
                        prod_mol), edit=edit, edit_atom=edit_atom)
                    prod_graph = MolGraph(mol=Chem.Mol(
                        int_mol), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
                    prod_tensors, prod_scopes = get_batch_graphs(
                        [prod_graph], use_rxn_class=self.use_rxn_class)
                    edits_prob, edits, edits_atom = [], [], []
                    edits_prob.extend(path['edits_prob'])
                    edits_prob.append(val)
                    edits.extend(path['edits'])
                    edits.append(edit)
                    edits_atom.extend(path['edits_atom'])
                    edits_atom.append(edit_atom)
                    new_path = {
                        'prod_mol': int_mol,
                        'steps': steps,
                        'prob': new_prob,
                        'edits_prob': edits_prob,
                        'tensors': prod_tensors,
                        'scopes': prod_scopes,
                        'state': state,
                        'state_scope': state_scope,
                        'edits': edits,
                        'edits_atom': edits_atom,
                        'finished': False,
                    }
                    new_paths.append(new_path)
                except:
                    continue

        return new_paths

    def get_top_k_paths(self, paths):
        k = min(len(paths), self.beam_size)
        path_argsort = np.argsort([-path['prob'] for path in paths])
        filtered_paths = [paths[i] for i in path_argsort[:k]]

        return filtered_paths

    def get_edit_from_logits(self, mol, edit_logits, idx, val):
        max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim

        if idx.item() == len(edit_logits) - 1:
            edit = 'Terminate'
            edit_atom = []

        elif idx.item() < max_bond_idx:
            bond_logits = edit_logits[:mol.GetNumBonds(
            ) * self.model.bond_outdim]
            bond_logits = bond_logits.reshape(
                mol.GetNumBonds(), self.model.bond_outdim)
            idx_tensor = torch.where(bond_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]

            bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
            a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
            a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

            a1, a2 = sorted([a1, a2])
            edit_atom = [a1, a2]
            edit = self.model.bond_vocab.get_elem(edit_idx)

        else:
            atom_logits = edit_logits[max_bond_idx:-1]

            assert len(atom_logits) == mol.GetNumAtoms() * \
                self.model.atom_outdim
            atom_logits = atom_logits.reshape(
                mol.GetNumAtoms(), self.model.atom_outdim)
            idx_tensor = torch.where(atom_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]
            atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

            a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
            edit_atom = a1
            edit = self.model.atom_vocab.get_elem(edit_idx)

        return edit, edit_atom

    def run_search(self, prod_smi: str, max_steps: int = 8, rxn_class: int = None) -> List[dict]:
        product = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(product)
        prod_graph = MolGraph(mol=Chem.Mol(
            product), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
        prod_tensors, prod_scopes = get_batch_graphs(
            [prod_graph], use_rxn_class=self.use_rxn_class)

        paths = []
        start_path = {
            'prod_mol': product,
            'steps': 0,
            'prob': 1.0,
            'edits_prob': [],
            'tensors': prod_tensors,
            'scopes': prod_scopes,
            'state': None,
            'state_scope': None,
            'edits': [],
            'edits_atom': [],
            'finished': False,
        }
        paths.append(start_path)

        for step_i in range(max_steps):
            followed_path = [path for path in paths if not path['finished']]
            if len(followed_path) == 0:
                break

            paths = [path for path in paths if path['finished']]

            for path in followed_path:
                new_paths = self.process_path(path, rxn_class)
                paths += new_paths

            paths = self.get_top_k_paths(paths)

            if all(path['finished'] for path in paths):
                break

        finished_paths = []
        for path in paths:
            if path['finished']:
                try:
                    int_mol = product
                    path['rxn_actions'] = []
                    for i, edit in enumerate(path['edits']):
                        if int_mol is None:
                            print("Interim mol is None")
                            break
                        if edit == 'Terminate':
                            edit_exe = Termination(action_vocab='Terminate')
                            path['rxn_actions'].append(edit_exe)
                            pred_mol = edit_exe.apply(int_mol)
                            [a.ClearProp('molAtomMapNumber')
                             for a in pred_mol.GetAtoms()]
                            pred_mol = Chem.MolFromSmiles(
                                Chem.MolToSmiles(pred_mol))
                            final_smi = Chem.MolToSmiles(pred_mol)
                            path['final_smi'] = final_smi

                        elif edit[0] == 'Change Atom':
                            edit_exe = AtomEditAction(
                                path['edits_atom'][i], *edit[1], action_vocab='Change Atom')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        elif edit[0] == 'Delete Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Delete Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        if edit[0] == 'Change Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Change Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        if edit[0] == 'Attaching LG':
                            edit_exe = AddGroupAction(
                                path['edits_atom'][i], edit[1], action_vocab='Attaching LG')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                    finished_paths.append(path)

                except Exception as e:
                    print(f'Exception while final mol to Smiles: {str(e)}')
                    path['final_smi'] = 'final_smi_unmapped'
                    finished_paths.append(path)

        return finished_paths
