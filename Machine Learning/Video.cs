using MediaToolkit;
using MediaToolkit.Model;
using MediaToolkit.Options;
using System;
using System.IO;

namespace ML
{
    public static class Video
    {
        public static void ExtractImages(string videoPath, string outImageDir)
        {
            MediaFile video = new MediaFile { Filename = videoPath };
            using (var engine = new Engine())
            {
                engine.GetMetadata(video);
                TimeSpan secPerFrame = TimeSpan.FromSeconds(1.0 / video.Metadata.VideoData.Fps);
                TimeSpan seek = TimeSpan.FromSeconds(0.0);
                int i = 0;
                while (seek <= video.Metadata.Duration)
                {
                    var options = new ConversionOptions { Seek = seek };
                    var outputFile = new MediaFile { Filename = $"{outImageDir}\\{i++}.jpg" };
                    engine.GetThumbnail(video, outputFile, options);
                    seek = seek.Add(secPerFrame);
                }
            }
        }
        public static void ExtractImage(string videoPath, string outImageDir, TimeSpan seek)
        {
            MediaFile video = new MediaFile { Filename = videoPath };
            string fileName = Path.GetFileNameWithoutExtension(videoPath);
            using (var engine = new Engine())
            {
                engine.GetMetadata(video);
                var options = new ConversionOptions { Seek = seek };
                var outputFile = new MediaFile { Filename = $"{outImageDir}\\{fileName}.jpg" };
                engine.GetThumbnail(video, outputFile, options);
            }
        }
    }
}
