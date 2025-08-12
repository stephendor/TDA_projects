"""Wrapper for the GNN models."""

from topobench.nn.wrappers.base import AbstractWrapper


class GNNWrapper(AbstractWrapper):
    r"""Wrapper for the GNN models.

    This wrapper defines the forward pass of the model. The GNN models return
    the embeddings of the cells of rank 0.
    """

    def forward(self, batch):
        r"""Forward pass for the GNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        x_0 = self.backbone(
            batch.x_0,
            batch.edge_index,
            edge_weight=batch.get("edge_weight", None),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
