using LiveCharts;
using LiveCharts.Geared;
using System;
using System.Windows;

namespace MachineLearningTester
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            Graph.DisableAnimations = true;
            Graph.AnimationsSpeed = TimeSpan.FromMilliseconds(50);
            Graph.Hoverable = false;

            GearedValues<double> gvalues1 = new GearedValues<double>();
            gvalues1.WithQuality(Quality.Low);
            GearedValues<double> gvalues2 = new GearedValues<double>();
            gvalues2.WithQuality(Quality.Low);
            GearedValues<double> gvalues3 = new GearedValues<double>();
            gvalues3.WithQuality(Quality.Low);
            GearedValues<double> gvalues4 = new GearedValues<double>();
            gvalues4.WithQuality(Quality.Low);

            SeriesCollection = new SeriesCollection
            {
                new GLineSeries
                {
                    Title = "Total Cost 1",
                    Values = gvalues1,
                    LineSmoothness = 1.0,
                },
                new GLineSeries
                {
                    Title = "Total Cost 2",
                    Values = gvalues2,
                    LineSmoothness = 1.0,
                },
                new GLineSeries
                {
                    Title = "Total Cost 3",
                    Values = gvalues3,
                    LineSmoothness = 1.0,
                },
                new GLineSeries
                {
                    Title = "Total Cost 4",
                    Values = gvalues4,
                    LineSmoothness = 1.0,
                },
            };
            DataContext = this;
            Loaded += MainWindow_Loaded;
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            Tests.CostChanged += Tests_CostChanged;
            await Tests.RunAll();
        }

        private void Tests_CostChanged(double oldCost, double newCost, int iteration)
        {
            int setIndex = iteration % 4;
            var values = SeriesCollection[setIndex].Values;
            values.Add(newCost);
            //if (values.Count > 5000)
            //    values.RemoveAt(0);
        }

        public SeriesCollection SeriesCollection { get; set; }
    }
}
