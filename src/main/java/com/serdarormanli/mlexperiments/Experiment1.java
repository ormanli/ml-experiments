package com.serdarormanli.mlexperiments;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;

public class Experiment1 {

    public static INDArray nonLin(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(new Exp(input.dup().neg())).add(1).rdiv(1);
    }

    public static INDArray nonLinDerivative(INDArray input) {
        return input.dup().rsub(1).mul(input.dup());
    }

    public static void main(String[] args) {
        INDArray x = Nd4j.create(new double[][]{{0, 0, 1},
                {0, 1, 1},
                {1, 0, 1},
                {1, 1, 1}});

        INDArray y = Nd4j.create(new double[]{0, 0, 1, 1}).transpose();

        INDArray syn0 = Nd4j.rand(x.columns(), y.columns()).mul(2).sub(1);

        for (int i = 0; i < 10000; i++) {
            INDArray l0 = x.dup();
            INDArray l1 = nonLin(l0.mmul(syn0));

            INDArray l1Error = y.sub(l1);

            INDArray l1Delta = l1Error.mul(nonLinDerivative(l1));

            syn0 = syn0.add(l0.transpose().mmul(l1Delta));

            System.out.println(l1.toString());
        }
    }
}
