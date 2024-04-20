from unittest import TestCase

from rdkit import Chem

from src.util.rdkit_util import remove_atom_map_numbers, standardize_building_block


class TestRDKitUtil(TestCase):
    def setUp(self):
        self.reactions_smiles_atom_mapped = (
            "F[B-](F)(F)[C:1](=[O:2])[c:15]1[cH:16][cH:18][cH:20][c:19]([Cl:21])["
            "n:17]1.O=C1OC2(CCCCC2)O[C:6]12O[NH:3][C@@H:4]1[C@H:5]2[CH2:22][N:23](["
            "C:25]([O:26][CH2:28][CH2:29][c:30]2[cH:31][cH:33][c:35]([O:37]["
            "CH3:39])[c:34]([O:36][CH3:38])[cH:32]2)=[O:27])[CH2:24]1.[NH2:7][c:8]1["
            "cH:9][cH:10][c:11]([F:40])[cH:12][c:13]1[SH:14]>>[C:1](=[O:2])([NH:3]["
            "C@@H:4]1[C@H:5]([c:6]2[n:7][c:8]3[cH:9][cH:10][c:11]([F:40])[cH:12]["
            "c:13]3[s:14]2)[CH2:22][N:23]([C:25]([O:26][CH2:28][CH2:29][c:30]2["
            "cH:31][cH:33][c:35]([O:37][CH3:39])[c:34]([O:36][CH3:38])[cH:32]2)=["
            "O:27])[CH2:24]1)[c:15]1[cH:16][cH:18][cH:20][c:19]([Cl:21])[n:17]1,"
        )
        self.correct_reactants_no_mapno = (
            "O=C(c1cccc(Cl)n1)[B-](F)(F)F",
            "COc1ccc(CCOC(=O)N2C[C@@H]3NOC4(OC5(CCCCC5)OC4=O)[C@@H]3C2)cc1OC",
            "Nc1ccc(F)cc1S",
        )

        self.reactants = (
            "O=C(c1cccc(Cl)n1)[B-](F)(F)F.[K+]",
            "COc1ccc(CCOC(=O)N2C[C@@H]3NO[C@]4(OC5(CCCCC5)OC4=O)[C@@H]3C2)cc1OC.Cl",
            "Nc1ccc(F)cc1S",
        )

        self.correct_standardized_reactants = (
            "O=C(c1cccc(Cl)n1)[B-](F)(F)F",
            "COc1ccc(CCOC(=O)N2C[C@@H]3NOC4(OC5(CCCCC5)OC4=O)[C@@H]3C2)cc1OC",
            "Nc1ccc(F)cc1S",
        )

    def test_standardize_building_blocks(self):
        for correct_smi, smi in zip(
            self.correct_standardized_reactants, self.reactants
        ):
            with self.subTest(smiles=smi):
                self.assertEqual(
                    Chem.MolToSmiles(standardize_building_block(smi)), correct_smi
                )

    def test_remove_atom_map_numbers(self):
        reactants = [
            Chem.MolFromSmiles(smi)
            for smi in self.reactions_smiles_atom_mapped.split(">>")[0].split(".")
        ]
        [remove_atom_map_numbers(mol) for mol in reactants]
        reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
        for correct_smi, smi in zip(self.correct_reactants_no_mapno, reactant_smiles):
            with self.subTest(smiles=smi):
                self.assertEqual(correct_smi, smi)

    def test_standardize_building_blocks_with_atom_map_no_removal(self):
        reactants = self.reactions_smiles_atom_mapped.split(">>")[0].split(".")
        reactant_smiles = [
            Chem.MolToSmiles(standardize_building_block(smi, remove_map_no=True))
            for smi in reactants
        ]
        for correct_smi, smi in zip(
            self.correct_standardized_reactants, reactant_smiles
        ):
            with self.subTest(smiles=smi):
                self.assertEqual(correct_smi, smi)
