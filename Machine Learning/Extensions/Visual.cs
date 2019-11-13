using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ML.Extensions
{
    public static class VisualExtensions
    {
        private static BitmapSource Capture(this Visual target, Rect? subAreaBounds = null)
        {
            if (target == null)
                return null;

            var dpi = VisualTreeHelper.GetDpi(target);
            Rect bounds = VisualTreeHelper.GetDescendantBounds(target);
            RenderTargetBitmap rtb = new RenderTargetBitmap(
                (int)(bounds.Width * dpi.PixelsPerInchX / 96.0),
                (int)(bounds.Height * dpi.PixelsPerInchY / 96.0),
                dpi.PixelsPerInchX,
                dpi.PixelsPerInchY,
                PixelFormats.Pbgra32);

            DrawingVisual dv = new DrawingVisual();
            using (DrawingContext ctx = dv.RenderOpen())
            {
                VisualBrush vb = new VisualBrush(target);
                if (subAreaBounds != null)
                    vb.Viewbox = subAreaBounds.Value;
                ctx.DrawRectangle(vb, null, new Rect(0, 0, bounds.Width, bounds.Height));
            }

            rtb.Render(dv);
            return rtb;
        }
    }
}
