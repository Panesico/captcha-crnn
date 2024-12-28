<?php
namespace KCAPTCHA;
require_once 'KCAPTCHA.php';

// Get command line arguments
$num_images = isset($argv[1]) ? (int)$argv[1] : 100;
$output_dir = isset($argv[2]) ? $argv[2] : 'captcha_imgs';
$offset = isset($argv[3]) ? (int)$argv[3] : 0; // Unique offset for filenames

// Ensure output directory exists
if (!file_exists($output_dir)) {
    mkdir($output_dir, 0777, true);
}

$results = array();

// Generate the captchas
for ($i = 0; $i < $num_images; $i++) {
    $captcha = new KCAPTCHA();
    $text = $captcha->getKeyString();
    $filename = sprintf('%s/captcha_%06d.jpg', $output_dir, $offset + $i);
    $img = $captcha->getImageResource();
    imagejpeg($img, $filename, 90);
    imagedestroy($img);
    $results[] = array(
        'filename' => $filename,
        'text' => $text
    );
    // Print progress
    fprintf(STDERR, "Generated image %d of %d (offset %d)\n", $i + 1, $num_images, $offset);
}

// Output results as JSON
echo json_encode($results, JSON_PRETTY_PRINT);
