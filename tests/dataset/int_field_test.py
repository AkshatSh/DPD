import unittest

import torch
from allennlp.common.checks import ConfigurationError

from dpd.dataset.fields import IntField


class IntFieldTest(unittest.TestCase):
    
    def test_constructor(self):
        '''
        Tests to make sure the IntField is being constructed properly
        '''
        field = IntField(200)
        assert field.number == 200

        # test invalid types
        self.assertRaises(ConfigurationError, lambda: IntField('string'))
        self.assertRaises(ConfigurationError, lambda: IntField(2.0))

    def test_empty_field(self):
        '''
        Tests to make sure the empty int field is constructed properly
        '''
        empty = IntField(1).empty_field()
        assert empty.number == -1
    
    def test_as_tensor(self):
        '''
        Tests to make sure that the field can be represented as a tensor
        '''
        field_val = 200
        field = IntField(field_val)

        field_tensor = field.as_tensor({})

        assert type(field_tensor) == torch.Tensor
        assert field_tensor.shape == (1,)
        assert field_tensor.item() == field_val
    
    def test_str(self):
        '''
        Tests to make sure that the field can be represented as a string
        '''
        field_val = 200
        field = IntField(field_val)

        field_str = str(field)

        assert field_str == "IntField with value: (200)."
    
    def test_eq(self):
        '''
        Tests to make sure that equivlance can be checked with int fields
        '''
        field_val = 200
        field = IntField(field_val)
        field_other = IntField(-field_val)

        assert field == 200
        assert field != 201
        assert field != -200
        assert field == field
        assert field != field_other


