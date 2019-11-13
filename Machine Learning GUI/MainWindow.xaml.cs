using LiveCharts;
using LiveCharts.Geared;
using Microsoft.Win32;
using Microsoft.WindowsAPICodePack.Shell;
using ML;
using System;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace MLUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            if (IsInDesignMode())
                return;

            //Graph.DisableAnimations = true;
            //Graph.AnimationsSpeed = TimeSpan.FromMilliseconds(50);
            //Graph.Hoverable = false;

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

            VideoControl.LoadedBehavior = MediaState.Manual;
            VideoControl.MediaOpened += VideoControl_MediaOpened;
            string dir = "C:\\Users\\dnedd\\Pictures\\ML\\";
            Media.Path = dir;
            SourceMediaPath = new Uri(dir + "test.mp4", UriKind.Absolute);

            imageCanvas.DataContext = this;
        }

        private void VideoControl_MediaOpened(object sender, RoutedEventArgs e)
        {
            _totalVideoLengthSpan = VideoControl.NaturalDuration.HasTimeSpan ? VideoControl.NaturalDuration.TimeSpan : TimeSpan.Zero;
            _totalVideoLengthStr = _totalVideoLengthSpan.ToString(@"hh\:mm\:ss");
        }
        public static bool IsInDesignMode()
        {
            using var process = Process.GetCurrentProcess();
            return process.ProcessName.Contains("devenv", StringComparison.InvariantCultureIgnoreCase);
        }

        private System.Windows.Point? _startPoint = null;
        private bool _isPlaying;
        private DispatcherTimer _timer = new DispatcherTimer();
        private string _totalVideoLengthStr = "00:00:00";
        private TimeSpan _totalVideoLengthSpan;

        private ObservableCollection<Uri> _imageCollection = new ObservableCollection<Uri>();
        public ObservableCollection<Uri> ImageCollection
        {
            get => _imageCollection;
            set => _imageCollection = value;
        }

        private static string[] VideoFormatExtensions => new string[]
        {
            "mp4",
            "mpg",
            "mpeg",
            "wmv",
            "mov",
            "gif",
            "webm",
            "flv",
            "mkv",
        };
        private static string[] ImageFormatExtensions => new string[]
        {
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "png",
            "tga",
            "bmp",
        };

        private Uri _sourceMediaPath;
        public Uri SourceMediaPath
        {
            get => _sourceMediaPath;
            set
            {
                _sourceMediaPath = value;

                string ext = System.IO.Path.GetExtension(_sourceMediaPath.LocalPath).Substring(1).ToLowerInvariant();
                IsSourceMediaImage = ImageFormatExtensions.Contains(ext);
                IsSourceMediaVideo = VideoFormatExtensions.Contains(ext);

                VideoControl.Source = _sourceMediaPath;
                if (IsSourceMediaVideo)
                {
                    IsPlaying = true;
                    KeyframePanelRowDef.Height = VideoPanelRowDef.Height = new GridLength(35, GridUnitType.Pixel);
                }
                else
                {
                    IsPlaying = false;
                    KeyframePanelRowDef.Height = VideoPanelRowDef.Height = new GridLength(0, GridUnitType.Pixel);
                }
            }
        }
        private bool _isSourceMediaImage = false;
        public bool IsSourceMediaImage
        {
            get => _isSourceMediaImage;
            set
            {
                _isSourceMediaImage = value;
            }
        }
        private bool _isSourceMediaVideo = false;
        public bool IsSourceMediaVideo
        {
            get => _isSourceMediaVideo;
            set
            {
                _isSourceMediaVideo = value;
            }
        }
        private bool _ended = false;
        public bool IsPlaying
        {
            get => _isPlaying;
            set
            {
                if (_isPlaying == value)
                    return;

                _isPlaying = value;

                if (_isPlaying)
                {
                    VideoControl.Play();
                    BtnPlay.Content = "Pause";
                    if (_ended)
                    {
                        VideoControl.Position = TimeSpan.FromMilliseconds(0);
                        _ended = false;
                    }
                    _timer.Start();
                }
                else
                {
                    if (!_ended)
                        VideoControl.Pause();
                    BtnPlay.Content = "Play";
                    _timer.Stop();
                }
                BtnSkipBackward.IsEnabled = BtnSkipForward.IsEnabled = !_isPlaying;
            }
        }
        private void VideoControl_MediaEnded(object sender, RoutedEventArgs e)
        {
            if (ChkLoop.IsChecked != true)
            {
                _ended = true;
                IsPlaying = false;
                return;
            }

            VideoControl.Position = TimeSpan.FromMilliseconds(0);
            VideoControl.Play();
        }
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            //Tests.CostChanged += Tests_CostChanged;
            //await Tests.RunAll();

            _timer.Tick += _timer_Tick;
            _timer.Interval = TimeSpan.FromMilliseconds(1);
        }

        private void _timer_Tick(object sender, EventArgs e)
        {
            TimeLabel.Content = VideoControl.Position.ToString(@"hh\:mm\:ss") + " / " + _totalVideoLengthStr;
            TimeIndicator.SetValue(Canvas.LeftProperty, VideoControl.Position.TotalSeconds / _totalVideoLengthSpan.TotalSeconds * KeyframeCanvas.ActualWidth);
        }

        private void NetworkCostChanged(double oldCost, double newCost, int iteration)
        {
            int setIndex = iteration % SeriesCollection.Count;
            var values = SeriesCollection[setIndex].Values;
            values.Add(newCost);
            //if (values.Count > 5000)
            //    values.RemoveAt(0);
        }

        public SeriesCollection SeriesCollection { get; set; }

        public void SetVideoPosition(double seconds)
        {
            VideoControl.Position = TimeSpan.FromSeconds(seconds);
        }
        private void ImageCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!(_startPoint is System.Windows.Point startPoint))
                return;

            System.Windows.Point localPoint = e.GetPosition(imageCanvas);

            if (localPoint.X < 0.0)
                localPoint.X = 0.0;
            if (localPoint.X > imageCanvas.ActualWidth)
                localPoint.X = imageCanvas.ActualWidth;

            if (localPoint.Y < 0.0)
                localPoint.Y = 0.0;
            if (localPoint.Y > imageCanvas.ActualHeight)
                localPoint.Y = imageCanvas.ActualHeight;

            double w = localPoint.X - startPoint.X;
            if (w < 0.0)
            {
                rect.SetValue(Canvas.LeftProperty, startPoint.X + w);
                rect.Width = -w;
            }
            else
                rect.Width = w;

            double h = localPoint.Y - startPoint.Y;
            if (h < 0.0)
            {
                rect.SetValue(Canvas.TopProperty, startPoint.Y + h);
                rect.Height = -h;
            }
            else
                rect.Height = h;
        }

        //private SolidColorBrush RectangleStroke { get; } = new SolidColorBrush(Color.FromArgb(128, 0, 0, 0));
        //private SolidColorBrush RectangleFill { get; } = new SolidColorBrush(Color.FromArgb(50, 128, 128, 128));
        private void ImageCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            imageCanvas.CaptureMouse();

            System.Windows.Point localPoint = e.GetPosition(imageCanvas);

            _startPoint = localPoint;

            rect.Width = 0.0;
            rect.Height = 0.0;
            rect.SetValue(Canvas.LeftProperty, localPoint.X);
            rect.SetValue(Canvas.TopProperty, localPoint.Y);
        }

        private void ImageCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            imageCanvas.ReleaseMouseCapture();
            _startPoint = null;
        }

        private void BtnSkipStart_Click(object sender, RoutedEventArgs e)
        {
            VideoControl.Position = TimeSpan.FromMilliseconds(0);
        }

        private void BtnSkipBackward_Click(object sender, RoutedEventArgs e)
        {

        }

        private void BtnPlay_Click(object sender, RoutedEventArgs e)
        {
            IsPlaying = !IsPlaying;
        }

        private void ChkLoop_Checked(object sender, RoutedEventArgs e)
        {

        }

        private void BtnSkipForward_Click(object sender, RoutedEventArgs e)
        {

        }

        private void BtnSkipEnd_Click(object sender, RoutedEventArgs e)
        {
            VideoControl.Position = VideoControl.NaturalDuration.TimeSpan;
        }

        private void BtnOpen_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "All Supported File Types(*.mp4,*.mov,*.mpeg,*.wmv,*.avi,*.mpg)|*.mp4;*.mov;*.mpeg;*.wmv;*.avi;*.mpg"
            };
            if ((bool)ofd.ShowDialog())
            {
                try
                {
                    using Stream checkStream = ofd.OpenFile();
                    if (checkStream != null)
                        SourceMediaPath = new Uri(ofd.FileName, UriKind.Absolute);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error: Could not read file from disk. Original error: " + ex.Message);
                }
            }
        }

        private void PhotosListBox_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            var media = (Media)PhotosListBox.SelectedItem;
            if (media.IsFile)
                SourceMediaPath = media.FilePathUri;
            else
                Media.Path = media.FilePath;
        }

        public MediaCollection Media { get; set; } = new MediaCollection();

        private void MenuItem_Click(object sender, RoutedEventArgs e)
        {

        }

        private void ImageCanvas_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            //System.Windows.Size prev = e.PreviousSize;
            //System.Windows.Size curr = e.NewSize;
            //double wDragRatio   = curr.Width    / prev.Width;
            //double hDragRatio   = curr.Height   / prev.Height;
            //double wDragDist    = curr.Width    - prev.Width;
            //double hDragDist    = curr.Height   - prev.Height;
            //double videoWidth   = VideoControl.NaturalVideoWidth;
            //double videoHeight  = VideoControl.NaturalVideoHeight;
            //double canvasWidth  = imageCanvas.ActualWidth;
            //double canvasHeight = imageCanvas.ActualHeight;

            //if (videoWidth / videoHeight > canvasWidth / canvasHeight)
            //{
            //    //TB padding

            //    rect.Width *= wDragRatio;

            //    double left = (double)rect.GetValue(Canvas.LeftProperty);
            //    rect.SetValue(Canvas.LeftProperty, left * wDragRatio);

            //    double top = (double)rect.GetValue(Canvas.TopProperty);
            //    rect.SetValue(Canvas.TopProperty, (top + hDragDist * 0.5));
            //}
            //else
            //{
            //    //LR padding

            //    rect.Height *= hDragRatio;
            //    rect.Width += hDragDist / 2;

            //    double top = (double)rect.GetValue(Canvas.TopProperty);
            //    rect.SetValue(Canvas.TopProperty, top * hDragRatio);

            //    double left = (double)rect.GetValue(Canvas.LeftProperty);
            //    rect.SetValue(Canvas.LeftProperty, (left + wDragDist * 0.5));
            //}
        }
    }
    /// <summary>
    /// This class describes a single photo - its location, the image and 
    /// the metadata extracted from the image.
    /// </summary>
    public class Media
    {
        public Media(string filePath)
        {
            FilePath = filePath;
            FilePathUri = new Uri(filePath);
            if (IsFile)
                Image = ShellFile.FromFilePath(filePath).Thumbnail.BitmapSource;
            else
            {
                Bitmap bitmap = DefaultIcons.FolderLarge.ToBitmap();
                IntPtr hBitmap = bitmap.GetHbitmap();
                Image = Imaging.CreateBitmapSourceFromHBitmap(
                    hBitmap, IntPtr.Zero, Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
            }
            Metadata = System.IO.Path.GetFileName(FilePath);
        }

        public bool IsFile => File.Exists(FilePath);
        public Uri FilePathUri { get; }
        public string FilePath { get; }
        public BitmapSource Image { get; set; }
        public string Metadata { get; }

        public override string ToString() => FilePathUri.ToString();
    }
    public static class DefaultIcons
    {
        private static readonly Lazy<Icon> _lazyFolderIcon = new Lazy<Icon>(FetchIcon, true);

        public static Icon FolderLarge
        {
            get { return _lazyFolderIcon.Value; }
        }

        private static Icon FetchIcon()
        {
            var tmpDir = Directory.CreateDirectory(System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString())).FullName;
            var icon = ExtractFromPath(tmpDir);
            Directory.Delete(tmpDir);
            return icon;
        }

        private static Icon ExtractFromPath(string path)
        {
            SHFILEINFO shinfo = new SHFILEINFO();
            SHGetFileInfo(
                path,
                0, ref shinfo, (uint)Marshal.SizeOf(shinfo),
                SHGFI_ICON | SHGFI_LARGEICON);
            return Icon.FromHandle(shinfo.hIcon);
        }

        //Struct used by SHGetFileInfo function
        [StructLayout(LayoutKind.Sequential)]
        private struct SHFILEINFO
        {
            public IntPtr hIcon;
            public int iIcon;
            public uint dwAttributes;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
            public string szDisplayName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 80)]
            public string szTypeName;
        };

        [DllImport("shell32.dll")]
        private static extern IntPtr SHGetFileInfo(string pszPath, uint dwFileAttributes, ref SHFILEINFO psfi, uint cbSizeFileInfo, uint uFlags);

        private const uint SHGFI_ICON = 0x100;
        private const uint SHGFI_LARGEICON = 0x0;
        private const uint SHGFI_SMALLICON = 0x000000001;
    }
    public class MediaCollection : ObservableCollection<Media>
    {
        public MediaCollection() { }
        public MediaCollection(string path) : this(new DirectoryInfo(path)) { }
        public MediaCollection(DirectoryInfo directory)
        {
            _directory = directory;
            Update();
        }

        private DirectoryInfo _directory;

        public string Path
        {
            set
            {
                _directory = new DirectoryInfo(value);
                Update();
            }
            get { return _directory.FullName; }
        }

        public DirectoryInfo Directory
        {
            set
            {
                _directory = value;
                Update();
            }
            get { return _directory; }
        }
        private void Update()
        {
            Clear();
            //try
            //{
                foreach (var f in _directory.GetFileSystemInfos(/*"*.jpg|*.png|*.tiff|*.tif|*.jpeg|*.bmp|*.tga|*.mov|*.mp4|*.mpg|*.mpeg|*.wmv|*.flv|*.mkv|*.gif"*/))
                    Add(new Media(f.FullName));

            //}
            //catch 
            //{
            //    MessageBox.Show("Directory not found.");
            //}
        }
    }
}
