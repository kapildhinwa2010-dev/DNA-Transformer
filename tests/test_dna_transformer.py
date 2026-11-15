import pytest
import torch
from dna_transformer import DNATransformer, create_dna_transformer

class TestDNATransformer:
    def test_init(self):
        m = DNATransformer(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNATransformer(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_transformer()
        assert isinstance(m, DNATransformer)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
