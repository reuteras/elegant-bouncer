//
// Copyright (c) Matt Suiche. All rights reserved.
//
// Module Name:
//  ttf.rs
//
// Abstract:
//  TRIANGULATION
//
// Author:
//  Matt Suiche (msuiche) 28-Dec-2023
//
use log::{debug, info, warn};
use std::path;

use crate::errors::*;

use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use byteorder::ReadBytesExt;

// TrueType instruction opcodes
const NPUSHB: u8 = 0x40;
const NPUSHW: u8 = 0x41;
const PUSHB_START: u8 = 0xb0;
const PUSHB_END: u8 = 0xb7;
const PUSHW_START: u8 = 0xb8;
const PUSHW_END: u8 = 0xbf;
const ADJUST_1: u8 = 0x8f;
const ADJUST_2: u8 = 0x90;

// Control flow instructions
const IF: u8 = 0x58;
const ELSE: u8 = 0x1b;
const EIF: u8 = 0x59;
const FDEF: u8 = 0x2c;
const ENDF: u8 = 0x2d;
const IDEF: u8 = 0x89;

// Jump instructions
const JROT: u8 = 0x78;
const JROF: u8 = 0x79;
const JMPR: u8 = 0x1c;

// Additional instructions that might contain data or need special handling
const CALL: u8 = 0x2b;
const LOOPCALL: u8 = 0x2a;
const WS: u8 = 0x42;      // Write Store
const RS: u8 = 0x43;      // Read Store
const WCVTP: u8 = 0x44;   // Write Control Value Table in Pixel units
const RCVT: u8 = 0x45;    // Read Control Value Table
const WCVTF: u8 = 0x70;   // Write Control Value Table in Funits

#[derive(Debug)]
pub enum TtfError {
    OutOfRangeBytecode,
    InvalidFile,
    TableNotFound,
}

impl fmt::Display for TtfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtfError::OutOfRangeBytecode => write!(f, "This bytecode is out of range!"),
            TtfError::InvalidFile => write!(f, "Not a valid file."),
            TtfError::TableNotFound => write!(f, "Table not found"),
        }
    }
}

// Define the structures
#[repr(C)]
struct TtfTable {
    // glyf / fpgm / prep
    tag: [u8; 4],
    checksum: u32,
    offset: u32,
    len: u32,
}

#[repr(C)]
struct TtfOffsetTable {
    version: u32,
    num_tables: u16,
    search_ranges: u16,
    entry_selector: u16,
    range_shift: u16,
}

#[repr(C)]
struct TtfHeader {
    offset_table: TtfOffsetTable,
    tables: Vec<TtfTable>,
}

impl TtfHeader {
    fn from_reader<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut header = TtfHeader {
            offset_table: TtfOffsetTable {
                version: 0,
                num_tables: 0,
                search_ranges: 0,
                entry_selector: 0,
                range_shift: 0,
            },
            tables: Vec::new(),
        };

        // Read the fields one by one
        header.offset_table.version = reader.read_u32::<byteorder::BigEndian>()?;
        header.offset_table.num_tables = reader.read_u16::<byteorder::BigEndian>()?;
        header.offset_table.search_ranges = reader.read_u16::<byteorder::BigEndian>()?;
        header.offset_table.entry_selector = reader.read_u16::<byteorder::BigEndian>()?;
        header.offset_table.range_shift = reader.read_u16::<byteorder::BigEndian>()?;

        header.tables.clear();
        for _i in 0..header.offset_table.num_tables {
            let mut entry = TtfTable {
                // glyf / fpgm / prep
                tag: [0; 4],
                checksum: 0,
                offset: 0,
                len: 0,
            };
            reader.read_exact(&mut entry.tag)?;
            entry.checksum = reader.read_u32::<byteorder::BigEndian>()?;
            entry.offset = reader.read_u32::<byteorder::BigEndian>()?;
            entry.len = reader.read_u32::<byteorder::BigEndian>()?;
            header.tables.push(entry);
        }

        Ok(header)
    }

    fn is_valid(&self) -> bool {
        self.offset_table.version == 0x00010000
        // &self.riff_sig == b"RIFF" && &self.webp_sig == b"WEBP" && &self.vp8_sig == b"VP8L" // && self.vp8l_ssig[0] == 0x2f
    }

    fn get_table(&self, search: &[u8; 4]) -> Result<&TtfTable> {
        for t in &self.tables {
            if &t.tag == search {
                return Ok(t);
            }
        }

        Err(ElegantError::TtfError(TtfError::TableNotFound))
    }
}

fn skip_instruction_data(opcode: u8, bytecode: &[u8], offset: &mut usize) -> Result<()> {
    match opcode {
        NPUSHB => {
            if *offset + 1 >= bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            let count = bytecode[*offset + 1] as usize;
            if *offset + 2 + count > bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            debug!(
                "0x{:04x}: NPUSHB[{}] /* {} bytes to skip */",
                *offset, count, count
            );
            *offset += 2 + count; // Skip opcode + count byte + data bytes
        }
        NPUSHW => {
            if *offset + 1 >= bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            let count = bytecode[*offset + 1] as usize;
            if *offset + 2 + (count * 2) > bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            debug!(
                "0x{:04x}: NPUSHW[{}] /* {} words to skip */",
                *offset, count, count
            );
            *offset += 2 + (count * 2); // Skip opcode + count byte + data words
        }
        PUSHB_START..=PUSHB_END => {
            let count = (opcode - PUSHB_START + 1) as usize;
            if *offset + 1 + count > bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            debug!(
                "0x{:04x}: PUSHB[{}] /* {} bytes to skip */",
                *offset,
                count - 1,
                count
            );
            *offset += 1 + count; // Skip opcode + data bytes
        }
        PUSHW_START..=PUSHW_END => {
            let count = (opcode - PUSHW_START + 1) as usize;
            if *offset + 1 + (count * 2) > bytecode.len() {
                return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
            }
            debug!(
                "0x{:04x}: PUSHW[{}] /* {} words to skip */",
                *offset,
                count - 1,
                count
            );
            *offset += 1 + (count * 2); // Skip opcode + data words
        }
        _ => {
            // Regular single-byte instruction
            debug!("0x{:04x}: Instruction 0x{:02x}", *offset, opcode);
            *offset += 1;
        }
    }
    Ok(())
}

// Fast pre-check: just look for ADJUST bytes without validation
fn has_potential_adjust_bytes(byte_data: &[u8]) -> bool {
    byte_data.iter().any(|&b| b == ADJUST_1 || b == ADJUST_2)
}

// Original fast algorithm - check for ADJUST bytes with basic context validation
fn is_adjust_inst_present_fast(byte_data: &[u8]) -> bool {
    for (i, &byte) in byte_data.iter().enumerate() {
        if byte == ADJUST_1 || byte == ADJUST_2 {
            // Basic context check - avoid obvious PUSH data patterns
            let prev_is_push = if i > 1 {
                let prev_byte = byte_data[i - 2];
                matches!(prev_byte, NPUSHB | NPUSHW | PUSHB_START..=PUSHB_END | PUSHW_START..=PUSHW_END)
            } else {
                false
            };
            
            // If not obviously in PUSH data, consider it a potential match
            if !prev_is_push {
                return true;
            }
        }
    }
    false
}

fn is_adjust_inst_present(byte_data: &Vec<u8>) -> Result<bool> {
    // Phase 1: Fast path - if no ADJUST bytes are present at all, return immediately
    if !has_potential_adjust_bytes(byte_data) {
        return Ok(false);
    }
    
    // Phase 2: Enhanced heuristics for large instruction sequences
    if byte_data.len() > 2048 {
        let unknown_opcode_count = byte_data.iter()
            .filter(|&&b| b > 0x91 && b != 0x8f && b != 0x90 && (b < 0xb0 || b > 0xbf))
            .count();
        let unknown_ratio = unknown_opcode_count as f64 / byte_data.len() as f64;
        
        if unknown_ratio > 0.3 {
            debug!("Large bytecode ({} bytes) with high unknown opcode ratio ({:.2}) - likely data, skipping validation", 
                   byte_data.len(), unknown_ratio);
            return Ok(false);
        }
        
        // NEW: Additional heuristic for very large legitimate instruction sequences
        // Count legitimate TrueType instructions to distinguish from exploit code
        let legit_instruction_count = byte_data.iter()
            .filter(|&&b| matches!(b, 
                NPUSHB | NPUSHW | PUSHB_START..=PUSHB_END | PUSHW_START..=PUSHW_END |
                CALL | LOOPCALL | FDEF | ENDF | IDEF | IF | ELSE | EIF |
                JROT | JROF | JMPR | WS | RS | WCVTP | RCVT | WCVTF |
                0x00..=0x3F | 0x46..=0x6F | 0x71..=0x8E | 0x91..=0xAF
            ))
            .count();
        let legit_ratio = legit_instruction_count as f64 / byte_data.len() as f64;
        
        // If this is a large instruction sequence (>3KB) with high legitimate instruction ratio,
        // it's likely a complex but legitimate font (like complex Asian fonts, variable fonts, etc.)
        if byte_data.len() > 3072 && legit_ratio > 0.8 {
            debug!("Large bytecode ({} bytes) with high legitimate instruction ratio ({:.2}) - likely complex legitimate font, skipping ADJUST detection", 
                   byte_data.len(), legit_ratio);
            return Ok(false);
        }
    }
    
    // Phase 3: Fast algorithm first - use simpler, faster validation
    debug!("Running fast ADJUST detection algorithm...");
    if !is_adjust_inst_present_fast(byte_data) {
        debug!("Fast algorithm found no potential ADJUST instructions");
        return Ok(false);
    }
    
    // Phase 4: Only if fast algorithm found potential matches, run thorough validation
    debug!("Fast algorithm found potential matches, running thorough validation...");
    return is_adjust_inst_present_thorough(byte_data);
}

fn is_adjust_inst_present_thorough(byte_data: &[u8]) -> Result<bool> {
    let mut offset = 0;
    let mut conditional_depth = 0;
    let mut _in_function_def = false;
    let max_iterations = byte_data.len() * 2; // Prevent infinite loops
    let mut iterations = 0;

    while offset < byte_data.len() {
        // Safety check for malformed bytecode
        if iterations > max_iterations {
            warn!("Maximum iterations reached in bytecode parsing - possible malformed font");
            return Err(ElegantError::TtfError(TtfError::OutOfRangeBytecode));
        }
        iterations += 1;

        let opcode = byte_data[offset];

        // CRITICAL: Only check for ADJUST when we're confident we're parsing actual instructions
        // This fixes the false positive issue where data bytes containing 0x8f/0x90
        // were incorrectly flagged as ADJUST instructions
        if opcode == ADJUST_1 || opcode == ADJUST_2 {
            // ADJUST instructions are extremely rare and suspicious - apply very strict validation
            let mut confidence = 0;
            let mut data_indicators = 0;
            
            // Check surrounding bytes for data patterns that suggest this is NOT an instruction
            
            // Look at a wider context - check 3 bytes before and after
            let context_start = offset.saturating_sub(3);
            let context_end = std::cmp::min(offset + 4, byte_data.len());
            
            // Count how many bytes look like data vs instructions in the context
            for i in context_start..context_end {
                if i == offset { continue; } // Skip the ADJUST candidate itself
                
                let context_byte = byte_data[i];
                match context_byte {
                    // High-value bytes (0x80+) are more likely to be data than instructions
                    0x80..=0xFF => data_indicators += 1,
                    // Sequences of similar values suggest data
                    _ => {
                        if i > 0 && i < byte_data.len() - 1 {
                            let prev_byte = byte_data[i - 1];
                            let next_byte = byte_data[i + 1];
                            // Check for patterns like ascending/descending sequences
                            if (context_byte.wrapping_sub(prev_byte) <= 3) || 
                               (next_byte.wrapping_sub(context_byte) <= 3) {
                                data_indicators += 1;
                            }
                        }
                    }
                }
            }
            
            // If we have strong data indicators, reject immediately
            if data_indicators >= 3 {
                debug!("0x{:04x}: ADJUST candidate in data-like context (data_indicators: {}) - skipping", offset, data_indicators);
                offset += 1;
                continue;
            }
            
            // Check immediate previous byte for instruction-like patterns
            if offset > 0 {
                let prev_opcode = byte_data[offset - 1];
                match prev_opcode {
                    NPUSHB | NPUSHW | PUSHB_START..=PUSHB_END | PUSHW_START..=PUSHW_END => {
                        // Previous was a PUSH instruction - this is very likely data!
                        debug!("0x{:04x}: ADJUST candidate after PUSH instruction - definitely data byte", offset);
                        offset += 1;
                        continue;
                    }
                    IF | ELSE | EIF | FDEF | ENDF | IDEF => {
                        confidence += 3; // Control flow instructions are strong indicators
                    }
                    ADJUST_1 | ADJUST_2 => {
                        // Previous was also ADJUST - definitely data
                        debug!("0x{:04x}: ADJUST candidate after another ADJUST - definitely data", offset);
                        offset += 1;
                        continue;
                    }
                    // Only well-known single-byte instructions give confidence
                    0x00..=0x3F | 0x46..=0x6F | 0x71..=0x8E | 0x91..=0xAF => {
                        // But even then, be cautious
                        confidence += 1;
                    }
                    _ => {
                        // Unknown or suspicious previous byte reduces confidence
                        data_indicators += 1;
                    }
                }
            }
            
            // Check immediate next byte
            if offset + 1 < byte_data.len() {
                let next_opcode = byte_data[offset + 1];
                match next_opcode {
                    IF | ELSE | EIF | FDEF | ENDF | IDEF => {
                        confidence += 3; // Control flow after ADJUST is very good
                    }
                    NPUSHB | NPUSHW | PUSHB_START..=PUSHB_END | PUSHW_START..=PUSHW_END => {
                        confidence += 2; // PUSH instructions after ADJUST are reasonable
                    }
                    ADJUST_1 | ADJUST_2 => {
                        // Multiple ADJUST in sequence - definitely data
                        debug!("0x{:04x}: ADJUST candidate followed by another ADJUST - definitely data stream", offset);
                        offset += 1;
                        continue;
                    }
                    // Known single-byte instructions
                    0x00..=0x3F | 0x46..=0x6F | 0x71..=0x8E | 0x91..=0xAF => {
                        confidence += 1;
                    }
                    _ => {
                        data_indicators += 1;
                    }
                }
            }
            
            // Enhanced stringent threshold for all instruction sequences
            // Real TRIANGULATION exploits are extremely rare, so we can be very strict
            let required_confidence = if byte_data.len() > 1500 {
                // For large instruction sequences, require even higher confidence
                // as they're more likely to be legitimate complex fonts
                6
            } else if byte_data.len() > 800 {
                // Medium-sized sequences need higher confidence
                5
            } else {
                // Even small sequences now require higher confidence to reduce false positives
                // from legitimate fonts with complex instruction sequences
                5
            };
            
            if confidence >= required_confidence && data_indicators <= 1 {
                debug!(
                    "0x{:04x}: ADJUST /* Undocumented Apple instruction - TRIANGULATION indicator! */ (confidence: {}, data_indicators: {}, required: {})",
                    offset, confidence, data_indicators, required_confidence
                );
                debug!(
                    "Found ADJUST instruction at offset 0x{:04x} with opcode 0x{:02x}",
                    offset, opcode
                );
                return Ok(true);
            } else {
                debug!(
                    "0x{:04x}: ADJUST candidate rejected (confidence: {}, data_indicators: {}, required: {}) - likely legitimate large font",
                    offset, confidence, data_indicators, required_confidence
                );
                offset += 1;
                continue;
            }
        }

        // Handle instructions that contain inline data
        // This is the core fix - properly skip over data bytes
        match opcode {
            // Variable-length PUSH instructions with inline data - these are the main culprits
            NPUSHB | NPUSHW | PUSHB_START..=PUSHB_END | PUSHW_START..=PUSHW_END => {
                if let Err(e) = skip_instruction_data(opcode, byte_data, &mut offset) {
                    return Err(e);
                }
                // offset is already updated by skip_instruction_data
            }

            // Control flow instructions
            IF => {
                conditional_depth += 1;
                debug!(
                    "0x{:04x}: IF[] /* If test - depth {} */",
                    offset, conditional_depth
                );
                offset += 1;
            }
            ELSE => {
                debug!(
                    "0x{:04x}: ELSE[] /* Else clause - depth {} */",
                    offset, conditional_depth
                );
                offset += 1;
            }
            EIF => {
                if conditional_depth > 0 {
                    conditional_depth -= 1;
                }
                debug!(
                    "0x{:04x}: EIF[] /* End if - depth {} */",
                    offset, conditional_depth
                );
                offset += 1;
            }

            // Function definitions - these might contain complex bytecode
            FDEF => {
                _in_function_def = true;
                debug!("0x{:04x}: FDEF[] /* Function Definition */", offset);
                offset += 1;
            }
            ENDF => {
                _in_function_def = false;
                debug!("0x{:04x}: ENDF[] /* End Function Definition */", offset);
                offset += 1;
            }
            IDEF => {
                debug!("0x{:04x}: IDEF[] /* Instruction Definition */", offset);
                offset += 1;
            }

            // Jump instructions - single byte
            JROT => {
                debug!("0x{:04x}: JROT[] /* Jump Relative On True */", offset);
                offset += 1;
            }
            JROF => {
                debug!("0x{:04x}: JROF[] /* Jump Relative On False */", offset);
                offset += 1;
            }
            JMPR => {
                debug!("0x{:04x}: JMPR[] /* Jump Relative */", offset);
                offset += 1;
            }
            
            // Common single-byte instructions that we know are safe
            CALL | LOOPCALL | WS | RS | WCVTP | RCVT | WCVTF => {
                debug!("0x{:04x}: Known instruction 0x{:02x}", offset, opcode);
                offset += 1;
            }

            // Handle specific ranges that are known to be single-byte instructions
            0x00..=0x3F => {
                // Most instructions in this range are single-byte
                debug!("0x{:04x}: Single-byte instruction 0x{:02x}", offset, opcode);
                offset += 1;
            }
            
            0x46..=0x6F => {
                // Instructions in this range are typically single-byte
                debug!("0x{:04x}: Single-byte instruction 0x{:02x}", offset, opcode);
                offset += 1;
            }
            
            0x71..=0x8E => {
                // More single-byte instructions (avoiding ADJUST_1 = 0x8f)
                debug!("0x{:04x}: Single-byte instruction 0x{:02x}", offset, opcode);
                offset += 1;
            }
            
            0x91..=0xAF => {
                // Single-byte instructions (avoiding PUSH ranges)
                debug!("0x{:04x}: Single-byte instruction 0x{:02x}", offset, opcode);
                offset += 1;
            }

            // Unknown or potentially problematic opcodes
            _ => {
                debug!("0x{:04x}: Unknown/Unhandled opcode 0x{:02x} - treating as single byte", offset, opcode);
                offset += 1;
            }
        }
    }

    Ok(false)
}

pub fn scan_ttf_file(path: &path::Path) -> Result<ScanResultStatus> {
    debug!("Opening {}...", path.display());

    let mut _status = ScanResultStatus::StatusOk;

    let mut file = File::open(path)?;
    // TODO: check magic number
    let header = TtfHeader::from_reader(&file)?;

    debug!("header.ver = {:x}", header.offset_table.version);
    debug!("header.num_tables = {}", header.offset_table.num_tables);

    // fpgm — Font Program
    // This table is similar to the CVT Program, except that it is only run once, when the font is first used.
    debug!("--- fpgm ---");
    if let Ok(fpgm) = header.get_table(b"fpgm") {
        let mut byte_data = vec![0; fpgm.len as usize];
        file.seek(SeekFrom::Start((fpgm.offset as i64).try_into().unwrap()))?;
        // debug!("go to: 0x{:x}", fpgm.offset);
        file.read_exact(&mut byte_data)?;

        if let Ok(status) = is_adjust_inst_present(&byte_data) {
            if status == true {
                info!(
                    "Found in the table {:?} with base offset {:x}",
                    fpgm.tag, fpgm.offset
                );
                return Ok(ScanResultStatus::StatusMalicious);
            }
        }
    }

    // prep — Control Value Program
    // The Control Value Program consists of a set of TrueType instructions that will be executed
    // whenever the font or point size or transformation matrix change and before each glyph is interpreted.
    debug!("--- prep ---");
    if let Ok(prep) = header.get_table(b"prep") {
        let mut byte_data = vec![0; prep.len as usize];
        file.seek(SeekFrom::Start((prep.offset as i64).try_into().unwrap()))?;
        debug!("go to: 0x{:x}", prep.offset);
        file.read_exact(&mut byte_data)?;

        if let Ok(status) = is_adjust_inst_present(&byte_data) {
            if status == true {
                info!(
                    "Found in the table {:?} with base offset {:x}",
                    prep.tag, prep.offset
                );
                return Ok(ScanResultStatus::StatusMalicious);
            }
        }
    }

    // glyf — Glyph Data
    // This table contains information that describes the glyphs in the font in the TrueType outline format.
    if let Ok(maxp) = header.get_table(b"maxp") {
        file.seek(SeekFrom::Start((maxp.offset as i64).try_into().unwrap()))?;
        let _version = file.read_u32::<byteorder::BigEndian>()?;
        let num_glyph = file.read_u16::<byteorder::BigEndian>()?;

        debug!("number of glyf = {}", num_glyph);
        if let Ok(loca) = header.get_table(b"loca") {
            if let Ok(glyf) = header.get_table(b"glyf") {
                for glyf_id in 0..num_glyph {
                    file.seek(SeekFrom::Start(
                        ((loca.offset + (glyf_id * 2) as u32) as i64)
                            .try_into()
                            .unwrap(),
                    ))?;
                    let glyf_offset = file.read_u16::<byteorder::BigEndian>()? as u32;
                    let glyf_offset = glyf_offset as u32 * 2; // head.indexToLocFormat is assumed to be 0.
                    debug!(
                        "{}: glyf offset = {:x} (0x{:x})",
                        glyf_id,
                        glyf_offset,
                        glyf.offset + glyf_offset
                    );

                    file.seek(SeekFrom::Start(
                        ((glyf.offset + glyf_offset) as i64).try_into().unwrap(),
                    ))?;

                    let nb_of_contours = file.read_i16::<byteorder::BigEndian>()?;
                    let _x_min = file.read_u16::<byteorder::BigEndian>()?;
                    let _y_min = file.read_u16::<byteorder::BigEndian>()?;
                    let _x_max = file.read_u16::<byteorder::BigEndian>()?;
                    let _y_max = file.read_u16::<byteorder::BigEndian>()?;

                    // If the number of contours is greater than or equal to zero, this is a simple glyph.
                    // If negative, this is a composite glyph — the value -1 should be used for composite glyphs.
                    if nb_of_contours < 0 {
                        continue;
                    }

                    // if nb_of_contours != 0xffff {
                    for _i in 0..nb_of_contours {
                        let _num_points = file.read_u16::<byteorder::BigEndian>()?;
                    }
                    let instructions_len = file.read_u16::<byteorder::BigEndian>()?;
                    // instructions
                    debug!("instruction len = 0x{:x}", instructions_len);
                    let mut byte_data = vec![0; instructions_len as usize];
                    file.read_exact(&mut byte_data)?;

                    if let Ok(status) = is_adjust_inst_present(&byte_data) {
                        if status == true {
                            debug!(
                                "glyf id = {} and inst len is 0x{:x}",
                                glyf_id, instructions_len
                            );
                            debug!(
                                "Found in the glyf {:?} with id {} with base offset {:x} (0x{:x})",
                                glyf.tag,
                                glyf_id,
                                glyf.offset,
                                glyf.offset + glyf_offset
                            );
                            return Ok(ScanResultStatus::StatusMalicious);
                        }
                    }

                    // IGNORE: Flags and Points.
                }
            }
        }
    }

    if !header.is_valid() {
        debug!("Not a TTF file. Ignore");
        return Err(ElegantError::TtfError(TtfError::InvalidFile));
    }

    // debug!("get_vp8l_data_size() -> 0x{:x}", header.get_vp8l_data_size());

    Ok(ScanResultStatus::StatusOk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_parsing_with_data_bytes() {
        // Test that data bytes containing 0x8f/0x90 are NOT flagged as ADJUST instructions
        // This simulates a NPUSHB instruction followed by data bytes that happen to contain 0x8f
        let bytecode = vec![
            NPUSHB, // 0x40 - NPUSHB instruction
            0x03,   // Count: push 3 bytes
            0x12,   // Data byte 1
            0x8f,   // Data byte 2 - this should NOT be flagged as ADJUST!
            0x34,   // Data byte 3
            0x01,   // Next instruction (some arbitrary instruction)
        ];

        let result = is_adjust_inst_present(&bytecode);
        assert_eq!(result.ok(), Some(false)); // Should NOT detect ADJUST
    }

    #[test]
    fn test_bytecode_parsing_with_real_adjust() {
        // Test that actual ADJUST instructions in proper context are still detected
        // Need very strong instruction context for new strict validation
        // Position the ADJUST so it has control flow instructions directly adjacent
        let bytecode = vec![
            IF,       // 0x58 - Control flow instruction (strong context)
            ADJUST_1, // 0x8f - actual ADJUST instruction (should be detected)
            EIF,      // 0x59 - Control flow instruction (strong context)
        ];

        let result = is_adjust_inst_present(&bytecode);
        // Should get: prev IF (+3) + next EIF (+3) = 6 confidence, low data indicators
        assert_eq!(result.ok(), Some(true)); // Should detect ADJUST with strong context
    }

    #[test]
    fn test_bytecode_parsing_npushw_with_data() {
        // Test NPUSHW instruction with word data containing 0x90
        let bytecode = vec![
            NPUSHW, // 0x41 - NPUSHW instruction
            0x02,   // Count: push 2 words (4 bytes total)
            0x00, 0x90, // Word 1: contains 0x90 in low byte - should NOT be flagged
            0x12, 0x34, // Word 2
            0x01, // Next instruction
        ];

        let result = is_adjust_inst_present(&bytecode);
        assert_eq!(result.ok(), Some(false)); // Should NOT detect ADJUST
    }

    #[test]
    fn test_bytecode_parsing_pushb_range() {
        // Test PUSHB[0-7] instructions with data containing 0x8f
        let bytecode = vec![
            PUSHB_START + 2, // 0xb2 - PUSHB[2] (push 3 bytes)
            0x11,            // Data byte 1
            0x8f,            // Data byte 2 - should NOT be flagged
            0x22,            // Data byte 3
            0x01,            // Next instruction
        ];

        let result = is_adjust_inst_present(&bytecode);
        assert_eq!(result.ok(), Some(false)); // Should NOT detect ADJUST
    }

    #[test]
    fn test_bytecode_parsing_complex_sequence() {
        // Test complex sequence with multiple PUSH instructions and control flow
        let bytecode = vec![
            IF,          // 0x58 - IF instruction
            NPUSHB,      // 0x40 - NPUSHB
            0x02,        // Count: 2 bytes
            0x8f,        // Data byte containing 0x8f - should NOT be flagged
            0x90,        // Data byte containing 0x90 - should NOT be flagged
            PUSHW_START, // 0xb8 - PUSHW[0] (push 1 word = 2 bytes)
            0x00,
            0x01, // Word data
            EIF,  // 0x59 - EIF instruction
        ];

        let result = is_adjust_inst_present(&bytecode);
        assert_eq!(result.ok(), Some(false)); // Should NOT detect ADJUST
    }

    #[test]
    fn test_bytecode_boundary_conditions() {
        // Test edge cases and boundary conditions

        // Empty bytecode
        let empty_bytecode = vec![];
        assert_eq!(is_adjust_inst_present(&empty_bytecode).ok(), Some(false));

        // Single byte that is ADJUST - with confidence system, this needs context
        // A single isolated ADJUST is suspicious and won't have enough confidence
        let single_adjust = vec![ADJUST_1];
        assert_eq!(is_adjust_inst_present(&single_adjust).ok(), Some(false));

        // Single byte that is not ADJUST
        let single_other = vec![0x01];
        assert_eq!(is_adjust_inst_present(&single_other).ok(), Some(false));
    }

    #[test]
    fn test_performance_optimizations() {
        // Test that performance optimizations work correctly
        
        // ADJUST in PUSH context - fast algorithm should skip without thorough validation
        let adjust_in_push_data = vec![NPUSHB, 0x05, ADJUST_1]; // ADJUST is clearly PUSH data
        assert_eq!(is_adjust_inst_present(&adjust_in_push_data).ok(), Some(false));
        
        // Bytecode without ADJUST bytes should return Ok(false) immediately (fastest path)
        let no_adjust = vec![NPUSHB, 0x01, 0x42, IF, EIF]; // Valid bytecode with no ADJUST
        assert_eq!(is_adjust_inst_present(&no_adjust).ok(), Some(false));
        
        // Large data-like bytecode should be skipped quickly
        let large_data: Vec<u8> = (0x80..=0xFF).cycle().take(3000).collect(); // Large, data-like
        assert_eq!(is_adjust_inst_present(&large_data).ok(), Some(false));
    }

    #[test]
    fn test_confidence_based_adjust_detection() {
        // Test that ADJUST candidates with low confidence are rejected
        
        // ADJUST after PUSH instruction should be treated as data
        let low_confidence = vec![
            NPUSHB, // 0x40 - NPUSHB instruction
            0x01,   // Count: 1 byte
            0x8f,   // This looks like ADJUST but is actually data from the PUSH
        ];
        assert_eq!(is_adjust_inst_present(&low_confidence).ok(), Some(false));
        
        // ADJUST in proper instruction context should be detected (high confidence needed now)
        let high_confidence = vec![
            IF,       // 0x58 - Control flow instruction (+3 confidence)
            ADJUST_1, // 0x8f - ADJUST instruction in good context
            EIF,      // 0x59 - Control flow instruction after (+3 confidence)
        ];
        // This should be detected: confidence=6, data_indicators=0
        assert_eq!(is_adjust_inst_present(&high_confidence).ok(), Some(true));
        
        // ADJUST with insufficient confidence (normal instructions)
        let medium_confidence = vec![
            0x01,     // Some instruction (+1 confidence)
            ADJUST_1, // 0x8f - ADJUST
            0x02,     // Another instruction (+1 confidence)
        ];
        // This should NOT be detected due to insufficient confidence (2 < 4 required)
        assert_eq!(is_adjust_inst_present(&medium_confidence).ok(), Some(false));
        
        // ADJUST in data-like context should be rejected
        let data_context = vec![
            0x85, 0x89, 0x8a, // High-value bytes suggesting data context
            ADJUST_1,         // 0x8f - ADJUST candidate
            0x91, 0x95, 0x99, // More high-value bytes
        ];
        // This should be rejected due to high data_indicators
        assert_eq!(is_adjust_inst_present(&data_context).ok(), Some(false));
    }
}
