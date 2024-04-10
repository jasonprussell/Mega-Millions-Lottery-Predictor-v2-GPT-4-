using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MegaMillionsPrediction
{
    // Define data structures
    public class MegaMillionsData
    {
        [LoadColumn(0)]
        public float DrawNumber { get; set; }

        [LoadColumn(1)]
        public float Ball1 { get; set; }

        [LoadColumn(2)]
        public float Ball2 { get; set; }

        [LoadColumn(3)]
        public float Ball3 { get; set; }

        [LoadColumn(4)]
        public float Ball4 { get; set; }

        [LoadColumn(5)]
        public float Ball5 { get; set; }

        [LoadColumn(6)]
        public float MegaBall { get; set; }
    }

    public class MegaMillionsPrediction
    {
        [ColumnName("Score")]
        public float Prediction { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Initialize ML.NET environment
            var mlContext = new MLContext();

            // Load data
            var dataPath = Path.Combine(Environment.CurrentDirectory, "MegaMillionsData.csv");
            var dataView = mlContext.Data.LoadFromTextFile<MegaMillionsData>(dataPath, hasHeader: true, separatorChar: ',');

            // Define training pipelines for each ball
            var ball1Pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ball1")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var ball2Pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ball2")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var ball3Pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ball3")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var ball4Pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ball4")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var ball5Pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Ball5")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var megaBallPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "MegaBall")
                .Append(mlContext.Transforms.Concatenate("Features", "DrawNumber"))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            // Train the models
            var ball1Model = ball1Pipeline.Fit(dataView);
            var ball2Model = ball2Pipeline.Fit(dataView);
            var ball3Model = ball3Pipeline.Fit(dataView);
            var ball4Model = ball4Pipeline.Fit(dataView);
            var ball5Model = ball5Pipeline.Fit(dataView);
            var megaBallModel = megaBallPipeline.Fit(dataView);

            // Make predictions
            var ball1PredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(ball1Model);
            var ball2PredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(ball2Model);
            var ball3PredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(ball3Model);
            var ball4PredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(ball4Model);
            var ball5PredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(ball5Model);
            var megaBallPredictionEngine = mlContext.Model.CreatePredictionEngine<MegaMillionsData, MegaMillionsPrediction>(megaBallModel);

            // Example input data (you can replace this with your own data)
            var testData = new MegaMillionsData()
            {
                DrawNumber = 2284, // Replace with the next draw number
            };

            // Clamp the predictions to the allowed ranges
            int Clamp(int value, int min, int max) => Math.Max(min, Math.Min(max, value));

            var ball1Prediction = Clamp((int)Math.Round(ball1PredictionEngine.Predict(testData).Prediction), 1, 70);
            var ball2Prediction = Clamp((int)Math.Round(ball2PredictionEngine.Predict(testData).Prediction), 1, 70);
            var ball3Prediction = Clamp((int)Math.Round(ball3PredictionEngine.Predict(testData).Prediction), 1, 70);
            var ball4Prediction = Clamp((int)Math.Round(ball4PredictionEngine.Predict(testData).Prediction), 1, 70);
            var ball5Prediction = Clamp((int)Math.Round(ball5PredictionEngine.Predict(testData).Prediction), 1, 70);
            var megaBallPrediction = Clamp((int)Math.Round(megaBallPredictionEngine.Predict(testData).Prediction), 1, 25);

            Console.WriteLine($"Predicted numbers for draw {testData.DrawNumber}:");
            Console.WriteLine($"Ball 1: {ball1Prediction}");
            Console.WriteLine($"Ball 2: {ball2Prediction}");
            Console.WriteLine($"Ball 3: {ball3Prediction}");
            Console.WriteLine($"Ball 4: {ball4Prediction}");
            Console.WriteLine($"Ball 5: {ball5Prediction}");
            Console.WriteLine($"Mega Ball: {megaBallPrediction}");
                      
        }
    }
}

