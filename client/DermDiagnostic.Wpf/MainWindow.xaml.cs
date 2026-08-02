using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace DermDiagnostic.Wpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml — Dermatology AI Desktop Client v2.0
    /// </summary>
    public partial class MainWindow : Window
    {
        private string? _currentImagePath;
        private CancellationTokenSource? _cts;

        private static readonly HttpClient _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(60)
        };

        // Diagnosis short-code extraction: e.g. "Melanoma (mel)" → "mel"
        private static string ExtractCode(string diagnosis)
        {
            int start = diagnosis.LastIndexOf('(');
            int end = diagnosis.LastIndexOf(')');
            if (start >= 0 && end > start)
                return diagnosis.Substring(start + 1, end - start - 1).ToUpper();
            return "?";
        }

        private static string GetMimeType(string path) =>
            Path.GetExtension(path).ToLowerInvariant() switch
            {
                ".png" => "image/png",
                ".bmp" => "image/bmp",
                _      => "image/jpeg"
            };

        public MainWindow()
        {
            InitializeComponent();
        }

        // -----------------------------------------------------------------------
        // Load image
        // -----------------------------------------------------------------------
        private void BtnLoad_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select a dermoscopic image",
                Filter = "Image files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|All files (*.*)|*.*"
            };

            if (dialog.ShowDialog() != true) return;

            _currentImagePath = dialog.FileName;
            ImgDisplay.Source = new BitmapImage(new Uri(_currentImagePath));
            TxtPlaceholder.Visibility = Visibility.Collapsed;
            // Fix #11 — OverlayCanvas is reserved for future bounding-box / lesion-boundary
            // annotation overlays drawn on top of the image (e.g. grad-CAM heatmaps).
            // Clearing it on each load ensures stale overlays from a previous image are removed.
            OverlayCanvas.Children.Clear();

            BtnAnalyze.IsEnabled = true;

            // Reset results panel
            TxtCode.Text = "—";
            TxtResult.Text = "Image loaded. Click 'Run AI Inference' to analyse.";
            TxtLatency.Text = "— ms";
            TxtConfidence.Text = "—";
            ConfidenceBar.Width = 0;
            TxtStatus.Text = $"Loaded: {Path.GetFileName(_currentImagePath)}";
        }

        // -----------------------------------------------------------------------
        // Run inference
        // -----------------------------------------------------------------------
        private async void BtnAnalyze_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_currentImagePath)) return;

            // Cancel any in-flight request
            _cts?.Cancel();
            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            SetAnalysingState(true);

            try
            {
                using var form = new MultipartFormDataContent();
                byte[] fileBytes = await File.ReadAllBytesAsync(_currentImagePath, token);
                var fileContent = new ByteArrayContent(fileBytes);
                fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse(GetMimeType(_currentImagePath));
                form.Add(fileContent, "file", Path.GetFileName(_currentImagePath));

                string? envUrl = (CmbEnvironment.SelectedItem as System.Windows.Controls.ComboBoxItem)?.Tag?.ToString();
                string targetUrl = (envUrl ?? "http://localhost:8000/").TrimEnd('/') + "/predict";

                HttpResponseMessage response = await _httpClient.PostAsync(targetUrl, form, token);

                if (!response.IsSuccessStatusCode)
                {
                    string errBody = await response.Content.ReadAsStringAsync(token);
                    ShowError($"Server returned {(int)response.StatusCode}: {errBody}");
                    return;
                }

                string json = await response.Content.ReadAsStringAsync(token);
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                string diagnosis   = root.GetProperty("diagnosis").GetString() ?? "Unknown";
                double confidence  = root.GetProperty("confidence").GetDouble();
                double latency     = root.GetProperty("processing_time_ms").GetDouble();

                // Update UI
                TxtCode.Text       = ExtractCode(diagnosis);
                TxtResult.Text     = diagnosis;
                TxtLatency.Text    = $"{latency:F1} ms";
                TxtConfidence.Text = $"{confidence * 100:F1}%";

                // Animate confidence bar (max width = sidebar 268px - padding)
                double maxBarWidth = ConfidenceBar.ActualWidth > 0
                    ? ((System.Windows.Controls.Grid)ConfidenceBar.Parent).ActualWidth
                    : 268;
                ConfidenceBar.Width = maxBarWidth * confidence;

                // Color-code confidence
                ConfidenceBar.Background = confidence >= 0.6
                    ? new SolidColorBrush(Color.FromRgb(34, 197, 94))   // green
                    : confidence >= 0.35
                        ? new SolidColorBrush(Color.FromRgb(234, 179, 8)) // yellow
                        : new SolidColorBrush(Color.FromRgb(239, 68, 68)); // red

                TxtStatus.Text = $"Analysis complete — {latency:F1} ms inference time";
            }
            catch (OperationCanceledException)
            {
                TxtStatus.Text = "Analysis cancelled.";
            }
            catch (HttpRequestException ex)
            {
                ShowError($"Cannot reach API.\nIf using Local Docker, ensure the container is running.\nIf using Cloud API, it may take 30s to wake up.\n\n{ex.Message}");
            }
            // M5 FIX — explicitly handle file system errors that occur when the image
            // is deleted, moved, or locked between the time the user selects it and
            // clicks Analyze.  The generic Exception catch below gives a cryptic message.
            catch (IOException ex)
            {
                ShowError($"Cannot read image file.\nThe file may have been moved, deleted, or is locked by another process.\n\nPath: {_currentImagePath}\n\n{ex.Message}");
            }
            catch (UnauthorizedAccessException ex)
            {
                ShowError($"Access denied when reading image file.\nEnsure you have permission to read this file.\n\nPath: {_currentImagePath}\n\n{ex.Message}");
            }
            catch (Exception ex)
            {
                ShowError($"Unexpected error: {ex.Message}");
            }
            finally
            {
                SetAnalysingState(false);
            }
        }

        // -----------------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------------
        private void SetAnalysingState(bool isAnalysing)
        {
            BtnAnalyze.IsEnabled    = !isAnalysing;
            PrgInference.Visibility = isAnalysing ? Visibility.Visible : Visibility.Collapsed;
            TxtStatus.Text          = isAnalysing ? "Analysing — please wait…" : TxtStatus.Text;
            TxtResult.Text          = isAnalysing ? "Running inference…" : TxtResult.Text;
        }

        private void ShowError(string message)
        {
            TxtResult.Text  = "Error — see details below";
            TxtStatus.Text  = "Analysis failed.";
            TxtCode.Text    = "ERR";
            MessageBox.Show(message, "Inference Error", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }
}
