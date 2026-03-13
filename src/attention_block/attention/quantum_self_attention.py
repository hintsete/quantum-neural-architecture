import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quantum.quantum_kernel import QuantumKernel


class QuantumSelfAttention(nn.Module):
    """
    Quantum Self-Attention Block with interference (phase) term.

    Args:
        n_qubits (int): Number of qubits in the quantum kernel
        n_layers (int): Number of layers in the ansatz
        device (qml.Device): PennyLane device for quantum circuits
        embed_dim (int): Dimension of input token embeddings
    """

    def __init__(self, n_qubits, n_layers, device, embed_dim, max_seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.max_seq_len = max_seq_len

        # Linear projection from embedding space to quantum feature dimension
        self.input_proj = nn.Linear(embed_dim, n_qubits)

        # Linear projection for value vectors
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum kernel module
        self.q_kernel = QuantumKernel(n_qubits, n_layers, device)

        # Output linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Ansatz parameters 
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

        # Interference parameter per token position
        
        self.phi = nn.Parameter(torch.randn(1, max_seq_len) * 0.1)

    def forward(self, X):
        """
        Forward pass of Quantum Self-Attention.

        Args:
            X (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim)
        Returns:
            out (torch.Tensor): Output embeddings, same shape as input
            attn_weights (torch.Tensor): Attention weight matrix, shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = X.shape
        assert seq_len <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # Project input embeddings to quantum feature space
        quantum_input = self.input_proj(X)  # (batch_size, seq_len, n_qubits)

        # Project input embeddings to value space
        V = self.value_proj(X)  # (batch_size, seq_len, embed_dim)

        attn_outputs = []
        attn_matrices = []

        for b in range(batch_size):
            # Compute quantum kernel / similarity matrix
            K = self.q_kernel.compute_kernel_matrix(quantum_input[b], self.theta)  

            # Compute interference term (phase differences)
            phi_b = self.phi[:, :seq_len] 
            phi_diff = phi_b.T - phi_b  
            K_interfered = K + torch.cos(phi_diff) 

            # Softmax over similarity + interference
            attn_weights = F.softmax(K_interfered, dim=-1)  

            # Weighted sum over values
            # attn_out = torch.matmul(attn_weights, V[b])  # (seq_len, embed_dim)
            attn_out = torch.matmul(attn_weights.to(V[b].dtype), V[b])  

            attn_outputs.append(attn_out)
            attn_matrices.append(attn_weights)

        # Stack batch dimension
        attn_outputs = torch.stack(attn_outputs)  
        attn_matrices = torch.stack(attn_matrices) 

        # Final linear projection
        out = self.out_proj(attn_outputs)

        return out, attn_matrices