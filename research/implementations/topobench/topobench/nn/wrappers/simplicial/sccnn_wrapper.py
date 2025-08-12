"""Wrapper for the SCCNN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class SCCNNWrapper(AbstractWrapper):
    r"""Wrapper for the SCCNN model.

    This wrapper defines the forward pass of the model. The SCCNN model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the SCCNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_all = (batch.x_0, batch.x_1, batch.x_2)
        laplacian_all = (
            batch.hodge_laplacian_0,
            batch.down_laplacian_1,
            batch.up_laplacian_1,
            batch.down_laplacian_2,
            batch.up_laplacian_2,
        )

        incidence_all = (batch.incidence_1, batch.incidence_2)
        x_0, x_1, x_2 = self.backbone(x_all, laplacian_all, incidence_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2

        return model_out
