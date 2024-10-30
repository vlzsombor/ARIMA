using System;
using System.Formats.Asn1;
using System.Security.Cryptography.X509Certificates;
using NumpyDotNet;
using Microsoft.VisualBasic;
using Microsoft.VisualBasic.FileIO;
using System.Runtime.CompilerServices;
using System.ComponentModel;

public class Program
{

    public static void Main(string[] args)
    {
        var A = np.array(new[,] { { 1, 2, 3 }, { 1, 2, 3 } });
        var W1 = new np.random();
        var b = W1.rand(new shape(3, 3)) - 0.5;
        var b1 = W1.rand(new shape(3, 3)) - 0.5;
        var b2 = W1.rand(new shape(3, 3)) - 0.5;

        Console.WriteLine("Hello, World!");
        System.Console.WriteLine(A);
        System.Console.WriteLine(b);
        System.Console.WriteLine(b1);
        System.Console.WriteLine(b2);

        var filePath = @"/Users/zsomborveres-lakos/git/ds/ARIMA/youtube/neural_network_from_scretch/digit-recognizer/train.csv";
        //var dataArray = np.array(File.ReadLines(filePath).Skip(1).Select(x => x.Split(',').Select(y => Int32.Parse(y))), ndmin: 2);
        var dataArray2 = File.ReadLines(filePath).Skip(1).Select(x => x.Split(',').Select(y => Int32.Parse(y)).ToArray()).ToArray();

        int[,] dataArray = new int[dataArray2.Length, dataArray2[0].Length];

        var y_test = new int[1000,1];
        var x_test = new int[1000,785];

        var y_train = new int[dataArray2.Length - 1000,1];
        var x_train = new int[dataArray2.Length - 1000,785];

        for (int i = 0; i < 1000; ++i)
        {
            y_test[i,0] = dataArray2[i][0];

            for (int j = 1; j < dataArray2[i].GetLength(0) ; j++)
            {
                x_test[i,j] = dataArray2[i][j];
            }
        }

        for (int i = 1000; i < dataArray2.GetLength(0); ++i)
        {
            y_train[i -1000,0] = dataArray2[i][0];

            for (int j = 1; j < dataArray2[i].GetLength(0); j++)
            {
                x_train[i-1000,j] = dataArray2[i][j];
            }
        }

        var neuron = new Neuron(dataArray, np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train));
        neuron.GradientDescent(50,0.1);
    }

}

public record Layer
{
    public ndarray W1 {get; set;}
    public ndarray b1 {get; set;}
    public ndarray W2 {get; set;}
    public ndarray b2 {get; set;}

    public Layer(ndarray W1, ndarray b1, ndarray W2, ndarray b2)
    {
        this.W1 = W1;
        this.b1 = b1;
        this.W2 = W2;
        this.b2 = b2;
    }
}
public record ForwardParameter(ndarray Z1, ndarray A1, ndarray Z2, ndarray A2);
public record BackParamParameter(ndarray dZ2, ndarray dW2, ndarray db2, ndarray dZ1, ndarray dW1, ndarray db1);

public class Neuron
{

    public ndarray Data { get; set; }

    public ndarray X_train { get; set; }
    public ndarray Y_train { get; set; }


    public ndarray X_dev { get; set; }
    public ndarray Y_dev { get; set; }


    private long n;
    private long m;

    public Neuron(int[,] dataArray, ndarray X_test,ndarray Y_test,ndarray X_train,ndarray Y_train)
    {
        Data = np.array(dataArray);

        this.X_train = X_train.T;
        this.Y_train = (ndarray)(Y_train.T)[0];

        this.m = Y_train.shape[0];
        this.n = Y_train.shape[1];

    }
    public Layer InitParam()
    {
        //W1, b1, W2, b2
        // W1 : pixel (780) x layer (10)
        // b1 : layer x 1
        // W2: W1.layer (10) x output (10)
        // b2: 10x1
        var random = new np.random();
        var W1 = random.rand(new shape(10, 785)) - 0.5;
        var b1 = random.rand(new shape(10, 1)) - 0.5;
        var W2 = random.rand(new shape(10, 10)) - 0.5;
        var b2 = random.rand(new shape(10, 1)) - 0.5;

        return new Layer(W1, b1, W2, b2);
    }

    public static ndarray Relu(ndarray input)
    {
        return np.maximum(0, input);
    }

    public static ndarray ReluDerivate(ndarray input)
    {
        return input > 0;
    }
    public static ndarray Softmax(ndarray input)
    {
        var res1 = np.exp(input);
        var res2 = np.exp(input);
        var res3 = np.sum(res2, 0);
        var res =  res1/res3;
        return res;
    }
    public ForwardParameter ForwardProp(Layer layer, ndarray X)
    {
        var Z1 = layer.W1.dot(X) + layer.b1;
        var A1 = Relu(Z1);
        var Z2 = layer.W2.dot(A1);
        var A2 = Softmax(Z2);
        return new ForwardParameter(Z1, A1, Z2, A2);
    }

    private ndarray OneHotEncode(ndarray Y)
    {
        var oneHotEncoding = np.zeros((10, 41000));
        // var oneHotEncoding = np.zeros((10, n));
        oneHotEncoding[Y, np.arange(n)] = 1;
        return oneHotEncoding;
    }

    public BackParamParameter BackProp(ForwardParameter forwardParameter, ndarray X, ndarray Y)
    {
        var oneHotEncoded = OneHotEncode(Y);
        var dZ2 = forwardParameter.A2 - oneHotEncoded;
        var dW2 = 1 / m * dZ2.dot(forwardParameter.A1.T);
        var db2 = 1 / m * np.sum(dZ2);
        var dZ1 = 1 / m * dZ2 * ReluDerivate(forwardParameter.Z1);
        var dW1 = 1 / m * dZ1.dot(X.T);
        var db1 = 1 / m * np.sum(dZ1);

        return new BackParamParameter(dZ2, dW2, db2, dZ1, dW1, db1);
    }
    public void UpdateParams(BackParamParameter backParamParameter, Layer layer, double alpha)
    {
        layer.W1 = layer.W1 - alpha * backParamParameter.dW1;
        layer.b1 = layer.b1 - alpha * backParamParameter.db1;
        layer.W2 = layer.W2 - alpha * backParamParameter.dW2;
        layer.b2 = layer.b2 - alpha * backParamParameter.db2;
    }

    public static ndarray GetPrediction(ndarray input)
    {
        return np.argmax(input, 0);
    }
    public static double GetAccuracy(ndarray prediction, ndarray Y)
    {
        var count = prediction.Intersect(Y).Count();
        return  count / Y.size;
    }

    public ForwardParameter? GradientDescent(int epoch,double alpha)
    {
        var layer = InitParam();
        ForwardParameter forwardParam = null;
        for (int i = 0; i < epoch; i++)
        {
            forwardParam = ForwardProp(layer, X_train);
            var b = BackProp(forwardParam, X_train, Y_train);
            UpdateParams(b, layer, alpha);
            if (i % 10 == 0)
            {
                var prediction = GetPrediction(forwardParam.A2);
                var s = prediction.shape;
                var accuracy = GetAccuracy(prediction, Y_train);
                System.Console.WriteLine($"Accuracy {accuracy}");
            }

        }
        return forwardParam;
    }
}
