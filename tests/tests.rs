use elegantbouncer::dng::scan_dng_file;
use elegantbouncer::jbig2::scan_pdf_jbig2_file;
use elegantbouncer::ttf::scan_ttf_file;
use elegantbouncer::webp::{
    is_code_lengths_count_valid, scan_webp_vp8l_file, MAX_DISTANCE_TABLE_SIZE,
};

use elegantbouncer::errors::ScanResultStatus;

use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_lengths_count() {
        let evil_array = [0, 1, 1, 1, 1, 1, 0, 0, 0, 11, 5, 1, 10, 4, 2, 2];
        let blastpass = is_code_lengths_count_valid(&evil_array.to_vec(), MAX_DISTANCE_TABLE_SIZE);

        assert!(blastpass);
    }

    #[test]
    fn test_blastpass_sample() {
        let path = Path::new("tests/samples/BLASTPASS.webp");
        let res = scan_webp_vp8l_file(path);

        assert_eq!(res.ok(), Some(ScanResultStatus::StatusMalicious));
    }

    #[test]
    fn test_blastpass_apple_sample() {
        let path = Path::new("tests/samples/replicatevalue_poc.not_.webp");
        let res = scan_webp_vp8l_file(path);

        assert_eq!(res.ok(), Some(ScanResultStatus::StatusMalicious));
    }

    #[test]
    fn test_forcedentry_sample() {
        let path = Path::new("tests/samples/FORCEDENTRY.gif");
        let res = scan_pdf_jbig2_file(path);

        assert_eq!(res.ok(), Some(ScanResultStatus::StatusMalicious));
    }

    #[test]
    fn test_run_ttf() {
        let path = Path::new("tests/samples/07558_CenturyGothic.ttf");
        let res = scan_ttf_file(path);

        assert_eq!(res.ok(), Some(ScanResultStatus::StatusOk));
    }

    // Note: These DNG tests reference local files that may not exist on all systems
    // Commenting out for now to avoid test failures
    // #[test]
    // fn test_cve_2025_43300_malicious() {
    //     let path = Path::new("/Users/msuiche/Downloads/IMGP0847_malicious.DNG");
    //     let res = scan_dng_file(path);
    //     assert_eq!(res, ScanResultStatus::StatusMalicious);
    // }

    // #[test]
    // fn test_cve_2025_43300_benign() {
    //     let path = Path::new("/Users/msuiche/Downloads/IMGP0847.DNG");
    //     let res = scan_dng_file(path);
    //     assert_eq!(res, ScanResultStatus::StatusOk);
    // }

    #[test]
    fn test_ttf_false_positive_fix() {
        // Test that legitimate fonts don't trigger false positives
        // CenturyGothic should be clean - it's a standard Windows font
        let path = Path::new("tests/samples/07558_CenturyGothic.ttf");
        let res = scan_ttf_file(path);

        // Should be OK (not malicious) after the false positive fix
        assert_eq!(res.ok(), Some(ScanResultStatus::StatusOk));
    }
}
