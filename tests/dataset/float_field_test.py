import unittest

import torch
import numpy as np
from allennlp.common.checks import ConfigurationError

from dpd.dataset.fields import FloatField


class FloatFieldTest(unittest.TestCase):
    
    def test_constructor(self):
        '''
        Tests to make sure the FloatField is being constructed properly
        '''
        field = FloatField(2.2)
        assert field.number == 2.2

        # test invalid types
        self.assertRaises(ConfigurationError, lambda: FloatField('string'))
        self.assertRaises(ConfigurationError, lambda: FloatField(2))

    def test_empty_field(self):
        '''
        Tests to make sure the empty int field is constructed properly
        '''
        empty = FloatField(1.0).empty_field()
        assert empty.number == -1.0
    
    def test_as_tensor(self):
        '''
        Tests to make sure that the field can be represented as a tensor
        '''
        field_val = 200.23
        field = FloatField(field_val)

        field_tensor = field.as_tensor({})

        assert type(field_tensor) == torch.Tensor
        assert field_tensor.shape == (1,)

        np.testing.assert_almost_equal(field_tensor.item(), field_val, decimal=1)
    
    def test_str(self):
        '''
        Tests to make sure that the field can be represented as a string
        '''
        field_val = 200.23
        field = FloatField(field_val)

        field_str = str(field)

        assert field_str == "FloatField with value: (200.23)."
    
    def test_eq(self):
        '''
        Tests to make sure that equivlance can be checked with int fields
        '''
        field_val = 200.23
        field = FloatField(field_val)
        field_other = FloatField(-field_val)

        assert field == 200.23
        assert field != 201
        assert field != -200.23
        assert field == field
        assert field != field_other


