using System;
using System.Formats.Asn1;
using System.Security.Cryptography.X509Certificates;
using NumpyDotNet;
using Microsoft.VisualBasic;
using Microsoft.VisualBasic.FileIO;
using System.Runtime.CompilerServices;

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

        var y_test = new int[1001,1];
        var x_test = new int[1001,785];

        var y_train = new int[dataArray2.Length - 1000+1,1];
        var x_train = new int[dataArray2.Length - 1000+1,785];

        for (int i = 0; i < 1000; i++)
        {
            y_test[i,0] = dataArray2[i][0];

            for (int j = 1; j < dataArray2[i].GetLength(0) ; j++)
            {
                x_test[i,j] = dataArray2[i][j];
            }
        }

        for (int i = 1000; i < dataArray2.GetLength(0); i++)
        {
            y_train[i -1000,0] = dataArray2[i][0];

            for (int j = 1; j < dataArray2[i].GetLength(0); j++)
            {
                x_train[i-1000,j] = dataArray2[i][j];
            }
        }

        var neuron = new Neuron(dataArray, np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train));
        neuron.GradientDescent(500);
    }



    static T[] GetSlice<T>(T[,] array, int row)
    {
        int cols = array.GetLength(1);
        var slice = new T[cols];

        for (int col = 0; col < cols; col++)
        {
            slice[col] = array[row, col];
        }

        return slice;
    }
}

public record class Layer(ndarray W1, ndarray b1, ndarray W2, ndarray b2);
public record class ForwardParameter(ndarray Z1, ndarray A1, ndarray Z2, ndarray A2);
public record class BackParamParameter(ndarray dZ2, ndarray dW2, ndarray db2, ndarray dZ1, ndarray dW1, ndarray db1);

public class Neuron
{

    public ndarray Data { get; set; }

    public ndarray X_train { get; set; }
    public ndarray Y_train { get; set; }


    public ndarray X_dev { get; set; }
    public ndarray Y_dev { get; set; }

    public Neuron(int[,] dataArray, ndarray X_test,ndarray Y_test,ndarray X_train,ndarray Y_train)
    {
        Data = np.array(dataArray);

        this.X_train = X_train.T;
        this.Y_train = Y_train.T;



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
        return np.exp(input) / np.sum(np.exp(input));
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
        var oneHotEncoding = np.zeros((10, Y.size));
        oneHotEncoding[Y, np.arange(Y.size)] = 1;
        return oneHotEncoding;
    }

    public BackParamParameter BackProp(ForwardParameter forwardParameter, ndarray X, ndarray Y)
    {
        var oneHotEncoded = OneHotEncode(Y);
        var dZ2 = forwardParameter.A2 - oneHotEncoded;
        var dW2 = 1 / Data.shape[0] * dZ2.dot(forwardParameter.A1);
        var db2 = 1 / Data.shape[0] * np.sum(dZ2);
        var dZ1 = 1 / Data.shape[0] * dZ2 * ReluDerivate(forwardParameter.Z1);
        var dW1 = 1 / Data.shape[0] * dZ1.dot(X);
        var db1 = 1 / Data.shape[0] * np.sum(dZ1);

        return new BackParamParameter(dZ2, dW2, db2, dZ1, dW1, db1);
    }
    public static ndarray GetPrediction(ndarray input)
    {
        return np.argmax(input, 0);
    }
    public static long GetAccuracy(ndarray prediction, ndarray Y)
    {
        return prediction.Count(x => x == Y) / Y.size;
    }

    public ForwardParameter? GradientDescent(int epoch)
    {
        var layer = InitParam();
        ForwardParameter forwardParam = null;
        for (int i = 0; i < epoch; i++)
        {
            forwardParam = ForwardProp(layer, X_train);
            var b = BackProp(forwardParam, X_train, Y_train);

            if (i % 10 == 0)
            {
                var prediction = GetPrediction(forwardParam.A2);
                var accuracy = GetAccuracy(prediction, Y_train);
                System.Console.WriteLine($"Accuracy {accuracy}");
            }

        }
        return forwardParam;
    }
}
