
// Copyright 2020 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#[cfg(test)]
mod psp_test_utils {
    use std::path::PathBuf;

    use ddo::*;

    use crate::{model::*, io_utils::read_instance};

    pub fn locate(id: &str) -> PathBuf {
        PathBuf::new()
            .join(env!("CARGO_MANIFEST_DIR"))
            .join("../resources/psp/")
            .join(id)
    }

    pub fn solve_id(id: &str) -> isize {
        let fname = locate(id);
        let fname = fname.to_str();
        let fname = fname.unwrap();
        
        let problem = read_instance(fname).unwrap();
        let relaxation = PspRelax::new(&problem);
        let ranking = PspRanking;

        let width = FixedWidth(1000);
        let cutoff = NoCutoff;
        let mut fringe = NoDupFrontier::new(MaxUB::new(&ranking));

        // This solver compile DD that allow the definition of long arcs spanning over several layers.
        let mut solver = DefaultSolver::<PspState, DefaultMDD<PspState>>::custom(
            &problem, 
            &relaxation, 
            &ranking, 
            &width, 
            &cutoff, 
            &mut fringe,
            1);

        let Completion { best_value , ..} = solver.maximize();
        best_value.map(|x| -x).unwrap_or(-1)
    }
}

#[cfg(test)]
mod tests_with_2_items {
    use super::psp_test_utils::solve_id;


    #[test]
    fn instance_with_2_items_1() {
        assert_eq!(13, solve_id("instancesWith2items/1"))
    }
    #[test]
    fn instance_with_2_items_2() {
        assert_eq!(54, solve_id("instancesWith2items/2"))
    }
    #[test]
    fn instance_with_2_items_3() {
        assert_eq!(46, solve_id("instancesWith2items/3"))
    }
    #[test]
    fn instance_with_2_items_4() {
        assert_eq!(2, solve_id("instancesWith2items/4"))
    }
    #[test]
    fn instance_with_2_items_5() {
        assert_eq!(78, solve_id("instancesWith2items/5"))
    }
    #[test]
    fn instance_with_2_items_6() {
        assert_eq!(52, solve_id("instancesWith2items/6"))
    }
    #[test]
    fn instance_with_2_items_7() {
        assert_eq!(255, solve_id("instancesWith2items/7"))
    }
    #[test]
    fn instance_with_2_items_8() {
        assert_eq!(168, solve_id("instancesWith2items/8"))
    }
    #[test]
    fn instance_with_2_items_9() {
        assert_eq!(120, solve_id("instancesWith2items/9"))
    }
    #[test]
    fn instance_with_2_items_10() {
        assert_eq!(695, solve_id("instancesWith2items/10"))
    }
    #[test]
    fn instance_with_2_items_11() {
        assert_eq!(125_002, solve_id("instancesWith2items/11"))
    }
    #[test]
    fn instance_with_2_items_12() {
        assert_eq!(120_013, solve_id("instancesWith2items/12"))
    }
    #[test]
    fn instance_with_2_items_13() {
        assert_eq!(750_008, solve_id("instancesWith2items/13"))
    }
    #[test]
    fn instance_with_2_items_14() {
        assert_eq!(1_250_005, solve_id("instancesWith2items/14"));
    }
}

/*
#[cfg(test)]
mod tests_with_5_items {
    use crate::psp_test_utils::solve_id;

    #[test]
    fn instance_with_5_items_1() {
        assert_eq!(1_377, solve_id("instancesWith5items/1"));
    }
    #[test]
    fn instance_with_5_items_2() {
        assert_eq!(1_447, solve_id("instancesWith5items/2"));
    }
    #[test]
    fn instance_with_5_items_3() {
        assert_eq!(1_107, solve_id("instancesWith5items/3"));
    }
    #[test]
    fn instance_with_5_items_4() {
        assert_eq!(1_182, solve_id("instancesWith5items/4"));
    }
    #[test]
    fn instance_with_5_items_5() {
        assert_eq!(1_471, solve_id("instancesWith5items/5"));
    }
    #[test]
    fn instance_with_5_items_6() {
        assert_eq!(1_386, solve_id("instancesWith5items/6"));
    }
    #[test]
    fn instance_with_5_items_7() {
        assert_eq!(1_382, solve_id("instancesWith5items/7"));
    }
    #[test]
    fn instance_with_5_items_8() {
        assert_eq!(3_117, solve_id("instancesWith5items/8"));
    }
    #[test]
    fn instance_with_5_items_9() {
        assert_eq!(1_315, solve_id("instancesWith5items/9"));
    }
    #[test]
    fn instance_with_5_items_10() {
        assert_eq!(1_952, solve_id("instancesWith5items/10"));
    }
    #[test]
    fn instance_with_5_items_11() {
        assert_eq!(1_202, solve_id("instancesWith5items/11"));
    }
    #[test]
    fn instance_with_5_items_12() {
        assert_eq!(1_135, solve_id("instancesWith5items/12"));
    }
    #[test]
    fn instance_with_5_items_13() {
        assert_eq!(1_026, solve_id("instancesWith5items/13"));
    }
    #[test]
    fn instance_with_5_items_14() {
        assert_eq!(1_363, solve_id("instancesWith5items/14"));
    }
    #[test]
    fn instance_with_5_items_15() {
        assert_eq!(1_430, solve_id("instancesWith5items/15"));
    }
    #[test]
    fn instance_with_5_items_16() {
        assert_eq!(1_145, solve_id("instancesWith5items/16"));
    }
    #[test]
    fn instance_with_5_items_17() {
        assert_eq!(1_367, solve_id("instancesWith5items/17"));
    }
    #[test]
    fn instance_with_5_items_18() {
        assert_eq!(1_315, solve_id("instancesWith5items/18"));
    }
    #[test]
    fn instance_with_5_items_19() {
        assert_eq!(1_490, solve_id("instancesWith5items/19"));
    }
    #[test]
    fn instance_with_5_items_20() {
        assert_eq!(779, solve_id("instancesWith5items/20"));
    }
    #[test]
    fn instance_with_5_items_21() {
        assert_eq!(2_774, solve_id("instancesWith5items/21"));
    }
    #[test]
    fn instance_with_5_items_22() {
        assert_eq!(1_397, solve_id("instancesWith5items/22"));
    }
    #[test]
    fn instance_with_5_items_23() {
        assert_eq!(1_473, solve_id("instancesWith5items/23"));
    }
    #[test]
    fn instance_with_5_items_24() {
        assert_eq!(1_308, solve_id("instancesWith5items/24"));
    }
    #[test]
    fn instance_with_5_items_25() {
        assert_eq!(1_810, solve_id("instancesWith5items/25"));
    }
    #[test]
    fn instance_with_5_items_26() {
        assert_eq!(1_753, solve_id("instancesWith5items/26"));
    }
    #[test]
    fn instance_with_5_items_27() {
        assert_eq!(1_277, solve_id("instancesWith5items/27"));
    }
    #[test]
    fn instance_with_5_items_28() {
        assert_eq!(1_036, solve_id("instancesWith5items/28"));
    }
    #[test]
    fn instance_with_5_items_29() {
        assert_eq!(880, solve_id("instancesWith5items/29"));
    }
    #[test]
    fn instance_with_5_items_30() {
        assert_eq!(588, solve_id("instancesWith5items/30"));
    }
    #[test]
    fn instance_with_5_items_31() {
        assert_eq!(2_249, solve_id("instancesWith5items/31"));
    }
    #[test]
    fn instance_with_5_items_32() {
        assert_eq!(1_562, solve_id("instancesWith5items/32"));
    }
    #[test]
    fn instance_with_5_items_33() {
        assert_eq!(1_104, solve_id("instancesWith5items/33"));
    }
    #[test]
    fn instance_with_5_items_34() {
        assert_eq!(1_108, solve_id("instancesWith5items/34"));
    }
    #[test]
    fn instance_with_5_items_35() {
        assert_eq!(2_655, solve_id("instancesWith5items/35"));
    }
    #[test]
    fn instance_with_5_items_36() {
        assert_eq!(1_493, solve_id("instancesWith5items/36"));
    }
    #[test]
    fn instance_with_5_items_37() {
        assert_eq!(1_840, solve_id("instancesWith5items/37"));
    }
    #[test]
    fn instance_with_5_items_38() {
        assert_eq!(1_113, solve_id("instancesWith5items/38"));
    }
    #[test]
    fn instance_with_5_items_39() {
        assert_eq!(1_744, solve_id("instancesWith5items/39"));
    }
    #[test]
    fn instance_with_5_items_40() {
        assert_eq!(1_193, solve_id("instancesWith5items/40"));
    }
    #[test]
    fn instance_with_5_items_41() {
        assert_eq!(1_189, solve_id("instancesWith5items/41"));
    }
    #[test]
    fn instance_with_5_items_42() {
        assert_eq!(996, solve_id("instancesWith5items/42"));
    }
    #[test]
    fn instance_with_5_items_43() {
        assert_eq!(1_995, solve_id("instancesWith5items/43"));
    }
    #[test]
    fn instance_with_5_items_44() {
        assert_eq!(1_339, solve_id("instancesWith5items/44"));
    }
    #[test]
    fn instance_with_5_items_45() {
        assert_eq!(1_182, solve_id("instancesWith5items/45"));
    }
    #[test]
    fn instance_with_5_items_46() {
        assert_eq!(1_575, solve_id("instancesWith5items/46"));
    }
    #[test]
    fn instance_with_5_items_47() {
        assert_eq!(1_371, solve_id("instancesWith5items/47"));
    }
    #[test]
    fn instance_with_5_items_48() {
        assert_eq!(1_572, solve_id("instancesWith5items/48"));
    }
    #[test]
    fn instance_with_5_items_49() {
        assert_eq!(1_882, solve_id("instancesWith5items/49"));
    }
    #[test]
    fn instance_with_5_items_50() {
        assert_eq!(1_405, solve_id("instancesWith5items/50"));
    }
    #[test]
    fn instance_with_5_items_51() {
        assert_eq!(1_397, solve_id("instancesWith5items/51"));
    }
    #[test]
    fn instance_with_5_items_52() {
        assert_eq!(1_531, solve_id("instancesWith5items/52"));
    }
    #[test]
    fn instance_with_5_items_53() {
        assert_eq!(1_108, solve_id("instancesWith5items/53"));
    }
    #[test]
    fn instance_with_5_items_54() {
        assert_eq!(1_056, solve_id("instancesWith5items/54"));
    }
    #[test]
    fn instance_with_5_items_55() {
        assert_eq!(1_015, solve_id("instancesWith5items/55"));
    }
    #[test]
    fn instance_with_5_items_56() {
        assert_eq!(2_121, solve_id("instancesWith5items/56"));
    }
    #[test]
    fn instance_with_5_items_57() {
        assert_eq!(919, solve_id("instancesWith5items/57"));
    }
    #[test]
    fn instance_with_5_items_58() {
        assert_eq!(1_384, solve_id("instancesWith5items/58"));
    }
    #[test]
    fn instance_with_5_items_59() {
        assert_eq!(1_490, solve_id("instancesWith5items/59"));
    }
    #[test]
    fn instance_with_5_items_60() {
        assert_eq!(1_265, solve_id("instancesWith5items/60"));
    }
    #[test]
    fn instance_with_5_items_61() {
        assert_eq!(977, solve_id("instancesWith5items/61"));
    }
    #[test]
    fn instance_with_5_items_62() {
        assert_eq!(872, solve_id("instancesWith5items/62"));
    }
    #[test]
    fn instance_with_5_items_63() {
        assert_eq!(1_481, solve_id("instancesWith5items/63"));
    }
    #[test]
    fn instance_with_5_items_64() {
        assert_eq!(1_869, solve_id("instancesWith5items/64"));
    }
    #[test]
    fn instance_with_5_items_65() {
        assert_eq!(1_781, solve_id("instancesWith5items/65"));
    }
    #[test]
    fn instance_with_5_items_66() {
        assert_eq!(1_571, solve_id("instancesWith5items/66"));
    }
    #[test]
    fn instance_with_5_items_67() {
        assert_eq!(1_185, solve_id("instancesWith5items/67"));
    }
    #[test]
    fn instance_with_5_items_68() {
        assert_eq!(1_131, solve_id("instancesWith5items/68"));
    }
    #[test]
    fn instance_with_5_items_69() {
        assert_eq!(1_619, solve_id("instancesWith5items/69"));
    }
    #[test]
    fn instance_with_5_items_70() {
        assert_eq!(1_148, solve_id("instancesWith5items/70"));
    }
    #[test]
    fn instance_with_5_items_71() {
        assert_eq!(1_024, solve_id("instancesWith5items/71"));
    }
    #[test]
    fn instance_with_5_items_72() {
        assert_eq!(1_555, solve_id("instancesWith5items/72"));
    }
    #[test]
    fn instance_with_5_items_73() {
        assert_eq!(1_104, solve_id("instancesWith5items/73"));
    }
    #[test]
    fn instance_with_5_items_74() {
        assert_eq!(1_053, solve_id("instancesWith5items/74"));
    }
    #[test]
    fn instance_with_5_items_75() {
        assert_eq!(1_021, solve_id("instancesWith5items/75"));
    }
    #[test]
    fn instance_with_5_items_76() {
        assert_eq!(1_484, solve_id("instancesWith5items/76"));
    }
    #[test]
    fn instance_with_5_items_77() {
        assert_eq!(852, solve_id("instancesWith5items/77"));
    }
    #[test]
    fn instance_with_5_items_78() {
        assert_eq!(1_297, solve_id("instancesWith5items/78"));
    }
    #[test]
    fn instance_with_5_items_79() {
        assert_eq!(1_555, solve_id("instancesWith5items/79"));
    }
    #[test]
    fn instance_with_5_items_80() {
        assert_eq!(1_382, solve_id("instancesWith5items/80"));
    }
    #[test]
    fn instance_with_5_items_81() {
        assert_eq!(1_889, solve_id("instancesWith5items/81"));
    }
    #[test]
    fn instance_with_5_items_82() {
        assert_eq!(1_947, solve_id("instancesWith5items/82"));
    }
    #[test]
    fn instance_with_5_items_83() {
        assert_eq!(1_681, solve_id("instancesWith5items/83"));
    }
    #[test]
    fn instance_with_5_items_84() {
        assert_eq!(728, solve_id("instancesWith5items/84"));
    }
    #[test]
    fn instance_with_5_items_85() {
        assert_eq!(2_113, solve_id("instancesWith5items/85"));
    }
    #[test]
    fn instance_with_5_items_86() {
        assert_eq!(886, solve_id("instancesWith5items/86"));
    }
    #[test]
    fn instance_with_5_items_87() {
        assert_eq!(1_152, solve_id("instancesWith5items/87"));
    }
    #[test]
    fn instance_with_5_items_88() {
        assert_eq!(1_353, solve_id("instancesWith5items/88"));
    }
    #[test]
    fn instance_with_5_items_89() {
        assert_eq!(1_390, solve_id("instancesWith5items/89"));
    }
    #[test]
    fn instance_with_5_items_90() {
        assert_eq!(2_449, solve_id("instancesWith5items/90"));
    }
    #[test]
    fn instance_with_5_items_91() {
        assert_eq!(998, solve_id("instancesWith5items/91"));
    }
    #[test]
    fn instance_with_5_items_92() {
        assert_eq!(1_453, solve_id("instancesWith5items/92"));
    }
    #[test]
    fn instance_with_5_items_93() {
        assert_eq!(1_286, solve_id("instancesWith5items/93"));
    }
    #[test]
    fn instance_with_5_items_94() {
        assert_eq!(1_403, solve_id("instancesWith5items/94"));
    }
    #[test]
    fn instance_with_5_items_95() {
        assert_eq!(1_618, solve_id("instancesWith5items/95"));
    }
    #[test]
    fn instance_with_5_items_96() {
        assert_eq!(1_409, solve_id("instancesWith5items/96"));
    }
    #[test]
    fn instance_with_5_items_97() {
        assert_eq!(1_290, solve_id("instancesWith5items/97"));
    }
    #[test]
    fn instance_with_5_items_98() {
        assert_eq!(1_403, solve_id("instancesWith5items/98"));
    }
    #[test]
    fn instance_with_5_items_99() {
        assert_eq!(1_237, solve_id("instancesWith5items/99"));
    }
    #[test]
    fn instance_with_5_items_100() {
        assert_eq!(1_040, solve_id("instancesWith5items/100"));
    }
    #[test]
    fn instance_with_5_items_101() {
        assert_eq!(5_429, solve_id("instancesWith5items/101"));
    }
    #[test]
    fn instance_with_5_items_102() {
        assert_eq!(6_563, solve_id("instancesWith5items/102"));
    }
    #[test]
    fn instance_with_5_items_103() {
        assert_eq!(10_981, solve_id("instancesWith5items/103"));
    }
    #[test]
    fn instance_with_5_items_104() {
        assert_eq!(4_891, solve_id("instancesWith5items/104"));
    }
    #[test]
    fn instance_with_5_items_105() {
        assert_eq!(7_282, solve_id("instancesWith5items/105"));
    }
    #[test]
    fn instance_with_5_items_106() {
        assert_eq!(10_275, solve_id("instancesWith5items/106"));
    }
    #[test]
    fn instance_with_5_items_107() {
        assert_eq!(7_666, solve_id("instancesWith5items/107"));
    }
    #[test]
    fn instance_with_5_items_108() {
        assert_eq!(8_743, solve_id("instancesWith5items/108"));
    }
    #[test]
    fn instance_with_5_items_109() {
        assert_eq!(9_093, solve_id("instancesWith5items/109"));
    }
    #[test]
    fn instance_with_5_items_110() {
        assert_eq!(9_094, solve_id("instancesWith5items/110"));
    }
    #[test]
    fn instance_with_5_items_111() {
        assert_eq!(7_677, solve_id("instancesWith5items/111"));
    }
    #[test]
    fn instance_with_5_items_112() {
        assert_eq!(11_903, solve_id("instancesWith5items/112"));
    }
    #[test]
    fn instance_with_5_items_113() {
        assert_eq!(7_440, solve_id("instancesWith5items/113"));
    }
    #[test]
    fn instance_with_5_items_114() {
        assert_eq!(12_260, solve_id("instancesWith5items/114"));
    }
    #[test]
    fn instance_with_5_items_115() {
        assert_eq!(8_539, solve_id("instancesWith5items/115"));
    }
    #[test]
    fn instance_with_5_items_116() {
        assert_eq!(10_370, solve_id("instancesWith5items/116"));
    }
    #[test]
    fn instance_with_5_items_117() {
        assert_eq!(11_163, solve_id("instancesWith5items/117"));
    }
    #[test]
    fn instance_with_5_items_118() {
        assert_eq!(12_975, solve_id("instancesWith5items/118"));
    }
    #[test]
    fn instance_with_5_items_119() {
        assert_eq!(10_253, solve_id("instancesWith5items/119"));
    }
    #[test]
    fn instance_with_5_items_120() {
        assert_eq!(7_218, solve_id("instancesWith5items/120"));
    }
    #[test]
    fn instance_with_5_items_121() {
        assert_eq!(8_489, solve_id("instancesWith5items/121"));
    }
    #[test]
    fn instance_with_5_items_122() {
        assert_eq!(11_055, solve_id("instancesWith5items/122"));
    }
    #[test]
    fn instance_with_5_items_123() {
        assert_eq!(7_610, solve_id("instancesWith5items/123"));
    }
    #[test]
    fn instance_with_5_items_124() {
        assert_eq!(8_124, solve_id("instancesWith5items/124"));
    }
    #[test]
    fn instance_with_5_items_125() {
        assert_eq!(6_630, solve_id("instancesWith5items/125"));
    }
    #[test]
    fn instance_with_5_items_126() {
        assert_eq!(11_389, solve_id("instancesWith5items/126"));
    }
    #[test]
    fn instance_with_5_items_127() {
        assert_eq!(8_057, solve_id("instancesWith5items/127"));
    }
    #[test]
    fn instance_with_5_items_128() {
        assert_eq!(8_032, solve_id("instancesWith5items/128"));
    }
    #[test]
    fn instance_with_5_items_129() {
        assert_eq!(10_595, solve_id("instancesWith5items/129"));
    }
    #[test]
    fn instance_with_5_items_130() {
        assert_eq!(5_107, solve_id("instancesWith5items/130"));
    }
    #[test]
    fn instance_with_5_items_131() {
        assert_eq!(10_922, solve_id("instancesWith5items/131"));
    }
    #[test]
    fn instance_with_5_items_132() {
        assert_eq!(11_947, solve_id("instancesWith5items/132"));
    }
    #[test]
    fn instance_with_5_items_133() {
        assert_eq!(12_599, solve_id("instancesWith5items/133"));
    }
    #[test]
    fn instance_with_5_items_134() {
        assert_eq!(8_219, solve_id("instancesWith5items/134"));
    }
    #[test]
    fn instance_with_5_items_135() {
        assert_eq!(10_401, solve_id("instancesWith5items/135"));
    }
    #[test]
    fn instance_with_5_items_136() {
        assert_eq!(14_507, solve_id("instancesWith5items/136"));
    }
    #[test]
    fn instance_with_5_items_137() {
        assert_eq!(11_938, solve_id("instancesWith5items/137"));
    }
    #[test]
    fn instance_with_5_items_138() {
        assert_eq!(17_869, solve_id("instancesWith5items/138"));
    }
    #[test]
    fn instance_with_5_items_139() {
        assert_eq!(9_848, solve_id("instancesWith5items/139"));
    }
    #[test]
    fn instance_with_5_items_140() {
        assert_eq!(9_211, solve_id("instancesWith5items/140"));
    }
    #[test]
    fn instance_with_5_items_141() {
        assert_eq!(10_919, solve_id("instancesWith5items/141"));
    }
    #[test]
    fn instance_with_5_items_142() {
        assert_eq!(5_191, solve_id("instancesWith5items/142"));
    }
    #[test]
    fn instance_with_5_items_143() {
        assert_eq!(11_659, solve_id("instancesWith5items/143"));
    }
    #[test]
    fn instance_with_5_items_144() {
        assert_eq!(8_065, solve_id("instancesWith5items/144"));
    }
    #[test]
    fn instance_with_5_items_145() {
        assert_eq!(6_860, solve_id("instancesWith5items/145"));
    }
    #[test]
    fn instance_with_5_items_146() {
        assert_eq!(5_894, solve_id("instancesWith5items/146"));
    }
    #[test]
    fn instance_with_5_items_147() {
        assert_eq!(7_522, solve_id("instancesWith5items/147"));
    }
    #[test]
    fn instance_with_5_items_148() {
        assert_eq!(6_185, solve_id("instancesWith5items/148"));
    }
    #[test]
    fn instance_with_5_items_149() {
        assert_eq!(10_126, solve_id("instancesWith5items/149"));
    }
    #[test]
    fn instance_with_5_items_150() {
        assert_eq!(11_118, solve_id("instancesWith5items/150"));
    }
    #[test]
    fn instance_with_5_items_151() {
        assert_eq!(7_459, solve_id("instancesWith5items/151"));
    }
    #[test]
    fn instance_with_5_items_152() {
        assert_eq!(11_425, solve_id("instancesWith5items/152"));
    }
    #[test]
    fn instance_with_5_items_153() {
        assert_eq!(12_906, solve_id("instancesWith5items/153"));
    }
    #[test]
    fn instance_with_5_items_154() {
        assert_eq!(7_597, solve_id("instancesWith5items/154"));
    }
    #[test]
    fn instance_with_5_items_155() {
        assert_eq!(7_151, solve_id("instancesWith5items/155"));
    }
    #[test]
    fn instance_with_5_items_156() {
        assert_eq!(7_642, solve_id("instancesWith5items/156"));
    }
    #[test]
    fn instance_with_5_items_157() {
        assert_eq!(8_469, solve_id("instancesWith5items/157"));
    }
    #[test]
    fn instance_with_5_items_158() {
        assert_eq!(5_030, solve_id("instancesWith5items/158"));
    }
    #[test]
    fn instance_with_5_items_159() {
        assert_eq!(8_274, solve_id("instancesWith5items/159"));
    }
    #[test]
    fn instance_with_5_items_160() {
        assert_eq!(8_624, solve_id("instancesWith5items/160"));
    }
    #[test]
    fn instance_with_5_items_161() {
        assert_eq!(6_391, solve_id("instancesWith5items/161"));
    }
    #[test]
    fn instance_with_5_items_162() {
        assert_eq!(5_074, solve_id("instancesWith5items/162"));
    }
    #[test]
    fn instance_with_5_items_163() {
        assert_eq!(8_790, solve_id("instancesWith5items/163"));
    }
    #[test]
    fn instance_with_5_items_164() {
        assert_eq!(12_475, solve_id("instancesWith5items/164"));
    }
    #[test]
    fn instance_with_5_items_165() {
        assert_eq!(11_804, solve_id("instancesWith5items/165"));
    }
    #[test]
    fn instance_with_5_items_166() {
        assert_eq!(10_948, solve_id("instancesWith5items/166"));
    }
    #[test]
    fn instance_with_5_items_167() {
        assert_eq!(10_770, solve_id("instancesWith5items/167"));
    }
    #[test]
    fn instance_with_5_items_168() {
        assert_eq!(6_541, solve_id("instancesWith5items/168"));
    }
    #[test]
    fn instance_with_5_items_169() {
        assert_eq!(7_315, solve_id("instancesWith5items/169"));
    }
    #[test]
    fn instance_with_5_items_170() {
        assert_eq!(8_658, solve_id("instancesWith5items/170"));
    }
    #[test]
    fn instance_with_5_items_171() {
        assert_eq!(4_488, solve_id("instancesWith5items/171"));
    }
    #[test]
    fn instance_with_5_items_172() {
        assert_eq!(8_548, solve_id("instancesWith5items/172"));
    }
    #[test]
    fn instance_with_5_items_173() {
        assert_eq!(10_797, solve_id("instancesWith5items/173"));
    }
    #[test]
    fn instance_with_5_items_174() {
        assert_eq!(9_055, solve_id("instancesWith5items/174"));
    }
    #[test]
    fn instance_with_5_items_175() {
        assert_eq!(16_288, solve_id("instancesWith5items/175"));
    }
    #[test]
    fn instance_with_5_items_176() {
        assert_eq!(7_897, solve_id("instancesWith5items/176"));
    }
    #[test]
    fn instance_with_5_items_177() {
        assert_eq!(7_870, solve_id("instancesWith5items/177"));
    }
    #[test]
    fn instance_with_5_items_178() {
        assert_eq!(9_056, solve_id("instancesWith5items/178"));
    }
    #[test]
    fn instance_with_5_items_179() {
        assert_eq!(7_989, solve_id("instancesWith5items/179"));
    }
    #[test]
    fn instance_with_5_items_180() {
        assert_eq!(10_545, solve_id("instancesWith5items/180"));
    }
    #[test]
    fn instance_with_5_items_181() {
        assert_eq!(6_917, solve_id("instancesWith5items/181"));
    }
    #[test]
    fn instance_with_5_items_182() {
        assert_eq!(11_612, solve_id("instancesWith5items/182"));
    }
    #[test]
    fn instance_with_5_items_183() {
        assert_eq!(18_138, solve_id("instancesWith5items/183"));
    }
    #[test]
    fn instance_with_5_items_184() {
        assert_eq!(8_638, solve_id("instancesWith5items/184"));
    }
    #[test]
    fn instance_with_5_items_185() {
        assert_eq!(5_001, solve_id("instancesWith5items/185"));
    }
    #[test]
    fn instance_with_5_items_186() {
        assert_eq!(10_392, solve_id("instancesWith5items/186"));
    }
    #[test]
    fn instance_with_5_items_187() {
        assert_eq!(8_285, solve_id("instancesWith5items/187"));
    }
    #[test]
    fn instance_with_5_items_188() {
        assert_eq!(9_066, solve_id("instancesWith5items/188"));
    }
    #[test]
    fn instance_with_5_items_189() {
        assert_eq!(7_322, solve_id("instancesWith5items/189"));
    }
    #[test]
    fn instance_with_5_items_190() {
        assert_eq!(10_461, solve_id("instancesWith5items/190"));
    }
    #[test]
    fn instance_with_5_items_191() {
        assert_eq!(15_996, solve_id("instancesWith5items/191"));
    }
    #[test]
    fn instance_with_5_items_192() {
        assert_eq!(7_329, solve_id("instancesWith5items/192"));
    }
    #[test]
    fn instance_with_5_items_193() {
        assert_eq!(6_322, solve_id("instancesWith5items/193"));
    }
    #[test]
    fn instance_with_5_items_194() {
        assert_eq!(7_599, solve_id("instancesWith5items/194"));
    }
    #[test]
    fn instance_with_5_items_195() {
        assert_eq!(6_923, solve_id("instancesWith5items/195"));
    }
    #[test]
    fn instance_with_5_items_196() {
        assert_eq!(6_350, solve_id("instancesWith5items/196"));
    }
    #[test]
    fn instance_with_5_items_197() {
        assert_eq!(7_813, solve_id("instancesWith5items/197"));
    }
    #[test]
    fn instance_with_5_items_198() {
        assert_eq!(10_465, solve_id("instancesWith5items/198"));
    }
    #[test]
    fn instance_with_5_items_199() {
        assert_eq!(13_770, solve_id("instancesWith5items/199"));
    }
    #[test]
    fn instance_with_5_items_200() {
        assert_eq!(11_403, solve_id("instancesWith5items/200"));
    }
    #[test]
    fn instance_with_5_items_201() {
        assert_eq!(6_362, solve_id("instancesWith5items/201"));
    }
    #[test]
    fn instance_with_5_items_202() {
        assert_eq!(32_267, solve_id("instancesWith5items/202"));
    }
    #[test]
    fn instance_with_5_items_203() {
        assert_eq!(15_888, solve_id("instancesWith5items/203"));
    }
    #[test]
    fn instance_with_5_items_204() {
        assert_eq!(26_694, solve_id("instancesWith5items/204"));
    }
    #[test]
    fn instance_with_5_items_205() {
        assert_eq!(27_217, solve_id("instancesWith5items/205"));
    }
    #[test]
    fn instance_with_5_items_206() {
        assert_eq!(23_729, solve_id("instancesWith5items/206"));
    }
    #[test]
    fn instance_with_5_items_207() {
        assert_eq!(38_932, solve_id("instancesWith5items/207"));
    }
    #[test]
    fn instance_with_5_items_208() {
        assert_eq!(55_894, solve_id("instancesWith5items/208"));
    }
    #[test]
    fn instance_with_5_items_209() {
        assert_eq!(24_527, solve_id("instancesWith5items/209"));
    }
    #[test]
    fn instance_with_5_items_210() {
        assert_eq!(25_177, solve_id("instancesWith5items/210"));
    }
    #[test]
    fn instance_with_5_items_211() {
        assert_eq!(50_781, solve_id("instancesWith5items/211"));
    }
    #[test]
    fn instance_with_5_items_212() {
        assert_eq!(22_418, solve_id("instancesWith5items/212"));
    }
    #[test]
    fn instance_with_5_items_213() {
        assert_eq!(38_549, solve_id("instancesWith5items/213"));
    }
    #[test]
    fn instance_with_5_items_214() {
        assert_eq!(36_160, solve_id("instancesWith5items/214"));
    }
    #[test]
    fn instance_with_5_items_215() {
        assert_eq!(34_286, solve_id("instancesWith5items/215"));
    }
    #[test]
    fn instance_with_5_items_216() {
        assert_eq!(29_755, solve_id("instancesWith5items/216"));
    }
    #[test]
    fn instance_with_5_items_217() {
        assert_eq!(25_820, solve_id("instancesWith5items/217"));
    }
    #[test]
    fn instance_with_5_items_218() {
        assert_eq!(30_081, solve_id("instancesWith5items/218"));
    }
    #[test]
    fn instance_with_5_items_219() {
        assert_eq!(28_448, solve_id("instancesWith5items/219"));
    }
    #[test]
    fn instance_with_5_items_220() {
        assert_eq!(39_945, solve_id("instancesWith5items/220"));
    }
    #[test]
    fn instance_with_5_items_221() {
        assert_eq!(18_215, solve_id("instancesWith5items/221"));
    }
    #[test]
    fn instance_with_5_items_222() {
        assert_eq!(24_725, solve_id("instancesWith5items/222"));
    }
    #[test]
    fn instance_with_5_items_223() {
        assert_eq!(24_647, solve_id("instancesWith5items/223"));
    }
    #[test]
    fn instance_with_5_items_224() {
        assert_eq!(24_404, solve_id("instancesWith5items/224"));
    }
    #[test]
    fn instance_with_5_items_225() {
        assert_eq!(44_980, solve_id("instancesWith5items/225"));
    }
    #[test]
    fn instance_with_5_items_226() {
        assert_eq!(30_008, solve_id("instancesWith5items/226"));
    }
    #[test]
    fn instance_with_5_items_227() {
        assert_eq!(55_772, solve_id("instancesWith5items/227"));
    }
    #[test]
    fn instance_with_5_items_228() {
        assert_eq!(24_632, solve_id("instancesWith5items/228"));
    }
    #[test]
    fn instance_with_5_items_229() {
        assert_eq!(23_605, solve_id("instancesWith5items/229"));
    }
    #[test]
    fn instance_with_5_items_230() {
        assert_eq!(46_021, solve_id("instancesWith5items/230"));
    }
    #[test]
    fn instance_with_5_items_231() {
        assert_eq!(30_743, solve_id("instancesWith5items/231"));
    }
    #[test]
    fn instance_with_5_items_232() {
        assert_eq!(36_657, solve_id("instancesWith5items/232"));
    }
    #[test]
    fn instance_with_5_items_233() {
        assert_eq!(25_098, solve_id("instancesWith5items/233"));
    }
    #[test]
    fn instance_with_5_items_234() {
        assert_eq!(26_242, solve_id("instancesWith5items/234"));
    }
    #[test]
    fn instance_with_5_items_235() {
        assert_eq!(23_788, solve_id("instancesWith5items/235"));
    }
    #[test]
    fn instance_with_5_items_236() {
        assert_eq!(25_377, solve_id("instancesWith5items/236"));
    }
    #[test]
    fn instance_with_5_items_237() {
        assert_eq!(22_574, solve_id("instancesWith5items/237"));
    }
    #[test]
    fn instance_with_5_items_238() {
        assert_eq!(53_752, solve_id("instancesWith5items/238"));
    }
    #[test]
    fn instance_with_5_items_239() {
        assert_eq!(40_154, solve_id("instancesWith5items/239"));
    }
    #[test]
    fn instance_with_5_items_240() {
        assert_eq!(40_535, solve_id("instancesWith5items/240"));
    }
    #[test]
    fn instance_with_5_items_241() {
        assert_eq!(19_831, solve_id("instancesWith5items/241"));
    }
    #[test]
    fn instance_with_5_items_242() {
        assert_eq!(34_807, solve_id("instancesWith5items/242"));
    }
    #[test]
    fn instance_with_5_items_243() {
        assert_eq!(22_057, solve_id("instancesWith5items/243"));
    }
    #[test]
    fn instance_with_5_items_244() {
        assert_eq!(59_796, solve_id("instancesWith5items/244"));
    }
    #[test]
    fn instance_with_5_items_245() {
        assert_eq!(42_257, solve_id("instancesWith5items/245"));
    }
    #[test]
    fn instance_with_5_items_246() {
        assert_eq!(22_163, solve_id("instancesWith5items/246"));
    }
    #[test]
    fn instance_with_5_items_247() {
        assert_eq!(21_689, solve_id("instancesWith5items/247"));
    }
    #[test]
    fn instance_with_5_items_248() {
        assert_eq!(27_037, solve_id("instancesWith5items/248"));
    }
    #[test]
    fn instance_with_5_items_249() {
        assert_eq!(41_563, solve_id("instancesWith5items/249"));
    }
    #[test]
    fn instance_with_5_items_250() {
        assert_eq!(36_005, solve_id("instancesWith5items/250"));
    }
    #[test]
    fn instance_with_5_items_251() {
        assert_eq!(31_740, solve_id("instancesWith5items/251"));
    }
    #[test]
    fn instance_with_5_items_252() {
        assert_eq!(34_609, solve_id("instancesWith5items/252"));
    }
    #[test]
    fn instance_with_5_items_253() {
        assert_eq!(26_397, solve_id("instancesWith5items/253"));
    }
    #[test]
    fn instance_with_5_items_254() {
        assert_eq!(29_085, solve_id("instancesWith5items/254"));
    }
    #[test]
    fn instance_with_5_items_255() {
        assert_eq!(19_406, solve_id("instancesWith5items/255"));
    }
    #[test]
    fn instance_with_5_items_256() {
        assert_eq!(52_140, solve_id("instancesWith5items/256"));
    }
    #[test]
    fn instance_with_5_items_257() {
        assert_eq!(51_098, solve_id("instancesWith5items/257"));
    }
    #[test]
    fn instance_with_5_items_258() {
        assert_eq!(28_809, solve_id("instancesWith5items/258"));
    }
    #[test]
    fn instance_with_5_items_259() {
        assert_eq!(22_482, solve_id("instancesWith5items/259"));
    }
    #[test]
    fn instance_with_5_items_260() {
        assert_eq!(27_245, solve_id("instancesWith5items/260"));
    }
    #[test]
    fn instance_with_5_items_261() {
        assert_eq!(29_940, solve_id("instancesWith5items/261"));
    }
    #[test]
    fn instance_with_5_items_262() {
        assert_eq!(26_317, solve_id("instancesWith5items/262"));
    }
    #[test]
    fn instance_with_5_items_263() {
        assert_eq!(44_309, solve_id("instancesWith5items/263"));
    }
    #[test]
    fn instance_with_5_items_264() {
        assert_eq!(20_224, solve_id("instancesWith5items/264"));
    }
    #[test]
    fn instance_with_5_items_265() {
        assert_eq!(26_862, solve_id("instancesWith5items/265"));
    }
    #[test]
    fn instance_with_5_items_266() {
        assert_eq!(18_177, solve_id("instancesWith5items/266"));
    }
    #[test]
    fn instance_with_5_items_267() {
        assert_eq!(22_395, solve_id("instancesWith5items/267"));
    }
    #[test]
    fn instance_with_5_items_268() {
        assert_eq!(36_252, solve_id("instancesWith5items/268"));
    }
    #[test]
    fn instance_with_5_items_269() {
        assert_eq!(30_745, solve_id("instancesWith5items/269"));
    }
    #[test]
    fn instance_with_5_items_270() {
        assert_eq!(44_229, solve_id("instancesWith5items/270"));
    }
    #[test]
    fn instance_with_5_items_271() {
        assert_eq!(41_741, solve_id("instancesWith5items/271"));
    }
    #[test]
    fn instance_with_5_items_272() {
        assert_eq!(32_932, solve_id("instancesWith5items/272"));
    }
    #[test]
    fn instance_with_5_items_273() {
        assert_eq!(34_862, solve_id("instancesWith5items/273"));
    }
    #[test]
    fn instance_with_5_items_274() {
        assert_eq!(49_822, solve_id("instancesWith5items/274"));
    }
    #[test]
    fn instance_with_5_items_275() {
        assert_eq!(22_666, solve_id("instancesWith5items/275"));
    }
    #[test]
    fn instance_with_5_items_276() {
        assert_eq!(35_113, solve_id("instancesWith5items/276"));
    }
    #[test]
    fn instance_with_5_items_277() {
        assert_eq!(24_186, solve_id("instancesWith5items/277"));
    }
    #[test]
    fn instance_with_5_items_278() {
        assert_eq!(56_268, solve_id("instancesWith5items/278"));
    }
    #[test]
    fn instance_with_5_items_279() {
        assert_eq!(31_354, solve_id("instancesWith5items/279"));
    }
    #[test]
    fn instance_with_5_items_280() {
        assert_eq!(47_218, solve_id("instancesWith5items/280"));
    }
    #[test]
    fn instance_with_5_items_281() {
        assert_eq!(30_840, solve_id("instancesWith5items/281"));
    }
    #[test]
    fn instance_with_5_items_282() {
        assert_eq!(20_090, solve_id("instancesWith5items/282"));
    }
    #[test]
    fn instance_with_5_items_283() {
        assert_eq!(40_288, solve_id("instancesWith5items/283"));
    }
    #[test]
    fn instance_with_5_items_284() {
        assert_eq!(61_788, solve_id("instancesWith5items/284"));
    }
    #[test]
    fn instance_with_5_items_285() {
        assert_eq!(29_512, solve_id("instancesWith5items/285"));
    }
    #[test]
    fn instance_with_5_items_286() {
        assert_eq!(27_838, solve_id("instancesWith5items/286"));
    }
    #[test]
    fn instance_with_5_items_287() {
        assert_eq!(17_778, solve_id("instancesWith5items/287"));
    }
    #[test]
    fn instance_with_5_items_288() {
        assert_eq!(24_490, solve_id("instancesWith5items/288"));
    }
    #[test]
    fn instance_with_5_items_289() {
        assert_eq!(68_772, solve_id("instancesWith5items/289"));
    }
    #[test]
    fn instance_with_5_items_290() {
        assert_eq!(31_503, solve_id("instancesWith5items/290"));
    }
    #[test]
    fn instance_with_5_items_291() {
        assert_eq!(22_673, solve_id("instancesWith5items/291"));
    }
    #[test]
    fn instance_with_5_items_292() {
        assert_eq!(20_747, solve_id("instancesWith5items/292"));
    }
    #[test]
    fn instance_with_5_items_293() {
        assert_eq!(30_363, solve_id("instancesWith5items/293"));
    }
    #[test]
    fn instance_with_5_items_294() {
        assert_eq!(47_437, solve_id("instancesWith5items/294"));
    }
    #[test]
    fn instance_with_5_items_295() {
        assert_eq!(34_123, solve_id("instancesWith5items/295"));
    }
    #[test]
    fn instance_with_5_items_296() {
        assert_eq!(39_994, solve_id("instancesWith5items/296"));
    }
    #[test]
    fn instance_with_5_items_297() {
        assert_eq!(24_221, solve_id("instancesWith5items/297"));
    }
    #[test]
    fn instance_with_5_items_298() {
        assert_eq!(35_050, solve_id("instancesWith5items/298"));
    }
    #[test]
    fn instance_with_5_items_299() {
        assert_eq!(25_643, solve_id("instancesWith5items/299"));
    }
    #[test]
    fn instance_with_5_items_300() {
        assert_eq!(32_054, solve_id("instancesWith5items/300"));
    }
    #[test]
    fn instance_with_5_items_301() {
        assert_eq!(28_689, solve_id("instancesWith5items/301"));
    }
    #[test]
    fn instance_with_5_items_302() {
        assert_eq!(19_983, solve_id("instancesWith5items/302"));
    }
    #[test]
    fn instance_with_5_items_303() {
        assert_eq!(14_971, solve_id("instancesWith5items/303"));
    }
    #[test]
    fn instance_with_5_items_304() {
        assert_eq!(21_127, solve_id("instancesWith5items/304"));
    }
    #[test]
    fn instance_with_5_items_305() {
        assert_eq!(27_065, solve_id("instancesWith5items/305"));
    }
    #[test]
    fn instance_with_5_items_306() {
        assert_eq!(18_182, solve_id("instancesWith5items/306"));
    }
    #[test]
    fn instance_with_5_items_307() {
        assert_eq!(25_555, solve_id("instancesWith5items/307"));
    }
    #[test]
    fn instance_with_5_items_308() {
        assert_eq!(22_138, solve_id("instancesWith5items/308"));
    }
    #[test]
    fn instance_with_5_items_309() {
        assert_eq!(24_977, solve_id("instancesWith5items/309"));
    }
    #[test]
    fn instance_with_5_items_310() {
        assert_eq!(19_827, solve_id("instancesWith5items/310"));
    }
    #[test]
    fn instance_with_5_items_311() {
        assert_eq!(24_891, solve_id("instancesWith5items/311"));
    }
    #[test]
    fn instance_with_5_items_312() {
        assert_eq!(17_809, solve_id("instancesWith5items/312"));
    }
    #[test]
    fn instance_with_5_items_313() {
        assert_eq!(21_763, solve_id("instancesWith5items/313"));
    }
    #[test]
    fn instance_with_5_items_314() {
        assert_eq!(15_904, solve_id("instancesWith5items/314"));
    }
    #[test]
    fn instance_with_5_items_315() {
        assert_eq!(24_636, solve_id("instancesWith5items/315"));
    }
    #[test]
    fn instance_with_5_items_316() {
        assert_eq!(23_509, solve_id("instancesWith5items/316"));
    }
    #[test]
    fn instance_with_5_items_317() {
        assert_eq!(19_560, solve_id("instancesWith5items/317"));
    }
    #[test]
    fn instance_with_5_items_318() {
        assert_eq!(17_820, solve_id("instancesWith5items/318"));
    }
    #[test]
    fn instance_with_5_items_319() {
        assert_eq!(24_075, solve_id("instancesWith5items/319"));
    }
    #[test]
    fn instance_with_5_items_320() {
        assert_eq!(21_347, solve_id("instancesWith5items/320"));
    }
    #[test]
    fn instance_with_5_items_321() {
        assert_eq!(22_634, solve_id("instancesWith5items/321"));
    }
    #[test]
    fn instance_with_5_items_322() {
        assert_eq!(17_299, solve_id("instancesWith5items/322"));
    }
    #[test]
    fn instance_with_5_items_323() {
        assert_eq!(22_818, solve_id("instancesWith5items/323"));
    }
    #[test]
    fn instance_with_5_items_324() {
        assert_eq!(17_205, solve_id("instancesWith5items/324"));
    }
    #[test]
    fn instance_with_5_items_325() {
        assert_eq!(17_317, solve_id("instancesWith5items/325"));
    }
    #[test]
    fn instance_with_5_items_326() {
        assert_eq!(19_558, solve_id("instancesWith5items/326"));
    }
    #[test]
    fn instance_with_5_items_327() {
        assert_eq!(19_522, solve_id("instancesWith5items/327"));
    }
    #[test]
    fn instance_with_5_items_328() {
        assert_eq!(19_356, solve_id("instancesWith5items/328"));
    }
    #[test]
    fn instance_with_5_items_329() {
        assert_eq!(18_483, solve_id("instancesWith5items/329"));
    }
    #[test]
    fn instance_with_5_items_330() {
        assert_eq!(17_502, solve_id("instancesWith5items/330"));
    }
    #[test]
    fn instance_with_5_items_331() {
        assert_eq!(22_938, solve_id("instancesWith5items/331"));
    }
    #[test]
    fn instance_with_5_items_332() {
        assert_eq!(15_383, solve_id("instancesWith5items/332"));
    }
    #[test]
    fn instance_with_5_items_333() {
        assert_eq!(17_076, solve_id("instancesWith5items/333"));
    }
    #[test]
    fn instance_with_5_items_334() {
        assert_eq!(16_593, solve_id("instancesWith5items/334"));
    }
    #[test]
    fn instance_with_5_items_335() {
        assert_eq!(17_547, solve_id("instancesWith5items/335"));
    }
    #[test]
    fn instance_with_5_items_336() {
        assert_eq!(16_081, solve_id("instancesWith5items/336"));
    }
    #[test]
    fn instance_with_5_items_337() {
        assert_eq!(23_464, solve_id("instancesWith5items/337"));
    }
    #[test]
    fn instance_with_5_items_338() {
        assert_eq!(24_392, solve_id("instancesWith5items/338"));
    }
    #[test]
    fn instance_with_5_items_339() {
        assert_eq!(20_859, solve_id("instancesWith5items/339"));
    }
    #[test]
    fn instance_with_5_items_340() {
        assert_eq!(24_040, solve_id("instancesWith5items/340"));
    }
    #[test]
    fn instance_with_5_items_341() {
        assert_eq!(17_642, solve_id("instancesWith5items/341"));
    }
    #[test]
    fn instance_with_5_items_342() {
        assert_eq!(29_174, solve_id("instancesWith5items/342"));
    }
    #[test]
    fn instance_with_5_items_343() {
        assert_eq!(16_958, solve_id("instancesWith5items/343"));
    }
    #[test]
    fn instance_with_5_items_344() {
        assert_eq!(24_779, solve_id("instancesWith5items/344"));
    }
    #[test]
    fn instance_with_5_items_345() {
        assert_eq!(17_807, solve_id("instancesWith5items/345"));
    }
    #[test]
    fn instance_with_5_items_346() {
        assert_eq!(23_217, solve_id("instancesWith5items/346"));
    }
    #[test]
    fn instance_with_5_items_347() {
        assert_eq!(17_112, solve_id("instancesWith5items/347"));
    }
    #[test]
    fn instance_with_5_items_348() {
        assert_eq!(18_304, solve_id("instancesWith5items/348"));
    }
    #[test]
    fn instance_with_5_items_349() {
        assert_eq!(19_252, solve_id("instancesWith5items/349"));
    }
    #[test]
    fn instance_with_5_items_350() {
        assert_eq!(20_963, solve_id("instancesWith5items/350"));
    }
    #[test]
    fn instance_with_5_items_351() {
        assert_eq!(18_431, solve_id("instancesWith5items/351"));
    }
    #[test]
    fn instance_with_5_items_352() {
        assert_eq!(14_548, solve_id("instancesWith5items/352"));
    }
    #[test]
    fn instance_with_5_items_353() {
        assert_eq!(24_097, solve_id("instancesWith5items/353"));
    }
    #[test]
    fn instance_with_5_items_354() {
        assert_eq!(18_397, solve_id("instancesWith5items/354"));
    }
    #[test]
    fn instance_with_5_items_355() {
        assert_eq!(20_082, solve_id("instancesWith5items/355"));
    }
    #[test]
    fn instance_with_5_items_356() {
        assert_eq!(25_951, solve_id("instancesWith5items/356"));
    }
    #[test]
    fn instance_with_5_items_357() {
        assert_eq!(24_711, solve_id("instancesWith5items/357"));
    }
    #[test]
    fn instance_with_5_items_358() {
        assert_eq!(15_540, solve_id("instancesWith5items/358"));
    }
    #[test]
    fn instance_with_5_items_359() {
        assert_eq!(23_544, solve_id("instancesWith5items/359"));
    }
    #[test]
    fn instance_with_5_items_360() {
        assert_eq!(25_576, solve_id("instancesWith5items/360"));
    }
    #[test]
    fn instance_with_5_items_361() {
        assert_eq!(20_178, solve_id("instancesWith5items/361"));
    }
    #[test]
    fn instance_with_5_items_362() {
        assert_eq!(25_660, solve_id("instancesWith5items/362"));
    }
    #[test]
    fn instance_with_5_items_363() {
        assert_eq!(21_718, solve_id("instancesWith5items/363"));
    }
    #[test]
    fn instance_with_5_items_364() {
        assert_eq!(13_451, solve_id("instancesWith5items/364"));
    }
    #[test]
    fn instance_with_5_items_365() {
        assert_eq!(18_924, solve_id("instancesWith5items/365"));
    }
    #[test]
    fn instance_with_5_items_366() {
        assert_eq!(17_329, solve_id("instancesWith5items/366"));
    }
    #[test]
    fn instance_with_5_items_367() {
        assert_eq!(20_495, solve_id("instancesWith5items/367"));
    }
    #[test]
    fn instance_with_5_items_368() {
        assert_eq!(22_470, solve_id("instancesWith5items/368"));
    }
    #[test]
    fn instance_with_5_items_369() {
        assert_eq!(17_026, solve_id("instancesWith5items/369"));
    }
    #[test]
    fn instance_with_5_items_370() {
        assert_eq!(20_612, solve_id("instancesWith5items/370"));
    }
    #[test]
    fn instance_with_5_items_371() {
        assert_eq!(22_561, solve_id("instancesWith5items/371"));
    }
    #[test]
    fn instance_with_5_items_372() {
        assert_eq!(18_555, solve_id("instancesWith5items/372"));
    }
    #[test]
    fn instance_with_5_items_373() {
        assert_eq!(21_115, solve_id("instancesWith5items/373"));
    }
    #[test]
    fn instance_with_5_items_374() {
        assert_eq!(20_897, solve_id("instancesWith5items/374"));
    }
    #[test]
    fn instance_with_5_items_375() {
        assert_eq!(16_320, solve_id("instancesWith5items/375"));
    }
    #[test]
    fn instance_with_5_items_376() {
        assert_eq!(21_676, solve_id("instancesWith5items/376"));
    }
    #[test]
    fn instance_with_5_items_377() {
        assert_eq!(17_923, solve_id("instancesWith5items/377"));
    }
    #[test]
    fn instance_with_5_items_378() {
        assert_eq!(19_197, solve_id("instancesWith5items/378"));
    }
    #[test]
    fn instance_with_5_items_379() {
        assert_eq!(21_823, solve_id("instancesWith5items/379"));
    }
    #[test]
    fn instance_with_5_items_380() {
        assert_eq!(21_060, solve_id("instancesWith5items/380"));
    }
    #[test]
    fn instance_with_5_items_381() {
        assert_eq!(15_800, solve_id("instancesWith5items/381"));
    }
    #[test]
    fn instance_with_5_items_382() {
        assert_eq!(19_632, solve_id("instancesWith5items/382"));
    }
    #[test]
    fn instance_with_5_items_383() {
        assert_eq!(22_752, solve_id("instancesWith5items/383"));
    }
    #[test]
    fn instance_with_5_items_384() {
        assert_eq!(24_638, solve_id("instancesWith5items/384"));
    }
    #[test]
    fn instance_with_5_items_385() {
        assert_eq!(17_237, solve_id("instancesWith5items/385"));
    }
    #[test]
    fn instance_with_5_items_386() {
        assert_eq!(24_513, solve_id("instancesWith5items/386"));
    }
    #[test]
    fn instance_with_5_items_387() {
        assert_eq!(11_593, solve_id("instancesWith5items/387"));
    }
    #[test]
    fn instance_with_5_items_388() {
        assert_eq!(24_005, solve_id("instancesWith5items/388"));
    }
    #[test]
    fn instance_with_5_items_389() {
        assert_eq!(21_467, solve_id("instancesWith5items/389"));
    }
    #[test]
    fn instance_with_5_items_390() {
        assert_eq!(21_009, solve_id("instancesWith5items/390"));
    }
    #[test]
    fn instance_with_5_items_391() {
        assert_eq!(19_102, solve_id("instancesWith5items/391"));
    }
    #[test]
    fn instance_with_5_items_392() {
        assert_eq!(16_101, solve_id("instancesWith5items/392"));
    }
    #[test]
    fn instance_with_5_items_393() {
        assert_eq!(21_403, solve_id("instancesWith5items/393"));
    }
    #[test]
    fn instance_with_5_items_394() {
        assert_eq!(22_079, solve_id("instancesWith5items/394"));
    }
    #[test]
    fn instance_with_5_items_395() {
        assert_eq!(28_479, solve_id("instancesWith5items/395"));
    }
    #[test]
    fn instance_with_5_items_396() {
        assert_eq!(27_675, solve_id("instancesWith5items/396"));
    }
    #[test]
    fn instance_with_5_items_397() {
        assert_eq!(14_936, solve_id("instancesWith5items/397"));
    }
    #[test]
    fn instance_with_5_items_398() {
        assert_eq!(20_488, solve_id("instancesWith5items/398"));
    }
    #[test]
    fn instance_with_5_items_399() {
        assert_eq!(17_721, solve_id("instancesWith5items/399"));
    }
    #[test]
    fn instance_with_5_items_400() {
        assert_eq!(23_637, solve_id("instancesWith5items/400"));
    }
    #[test]
    fn instance_with_5_items_401() {
        assert_eq!(21_194, solve_id("instancesWith5items/401"));
    }
    #[test]
    fn instance_with_5_items_402() {
        assert_eq!(21_258, solve_id("instancesWith5items/402"));
    }
    #[test]
    fn instance_with_5_items_403() {
        assert_eq!(16_335, solve_id("instancesWith5items/403"));
    }
    #[test]
    fn instance_with_5_items_404() {
        assert_eq!(22_152, solve_id("instancesWith5items/404"));
    }
    #[test]
    fn instance_with_5_items_405() {
        assert_eq!(28_100, solve_id("instancesWith5items/405"));
    }
    #[test]
    fn instance_with_5_items_406() {
        assert_eq!(19_482, solve_id("instancesWith5items/406"));
    }
    #[test]
    fn instance_with_5_items_407() {
        assert_eq!(26_222, solve_id("instancesWith5items/407"));
    }
    #[test]
    fn instance_with_5_items_408() {
        assert_eq!(23_265, solve_id("instancesWith5items/408"));
    }
    #[test]
    fn instance_with_5_items_409() {
        assert_eq!(25_862, solve_id("instancesWith5items/409"));
    }
    #[test]
    fn instance_with_5_items_410() {
        assert_eq!(21_058, solve_id("instancesWith5items/410"));
    }
    #[test]
    fn instance_with_5_items_411() {
        assert_eq!(25_751, solve_id("instancesWith5items/411"));
    }
    #[test]
    fn instance_with_5_items_412() {
        assert_eq!(18_839, solve_id("instancesWith5items/412"));
    }
    #[test]
    fn instance_with_5_items_413() {
        assert_eq!(22_696, solve_id("instancesWith5items/413"));
    }
    #[test]
    fn instance_with_5_items_414() {
        assert_eq!(16_803, solve_id("instancesWith5items/414"));
    }
    #[test]
    fn instance_with_5_items_415() {
        assert_eq!(25_349, solve_id("instancesWith5items/415"));
    }
    #[test]
    fn instance_with_5_items_416() {
        assert_eq!(24_557, solve_id("instancesWith5items/416"));
    }
    #[test]
    fn instance_with_5_items_417() {
        assert_eq!(20_596, solve_id("instancesWith5items/417"));
    }
    #[test]
    fn instance_with_5_items_418() {
        assert_eq!(19_116, solve_id("instancesWith5items/418"));
    }
    #[test]
    fn instance_with_5_items_419() {
        assert_eq!(25_088, solve_id("instancesWith5items/419"));
    }
    #[test]
    fn instance_with_5_items_420() {
        assert_eq!(22_194, solve_id("instancesWith5items/420"));
    }
    #[test]
    fn instance_with_5_items_421() {
        assert_eq!(23_423, solve_id("instancesWith5items/421"));
    }
    #[test]
    fn instance_with_5_items_422() {
        assert_eq!(18_650, solve_id("instancesWith5items/422"));
    }
    #[test]
    fn instance_with_5_items_423() {
        assert_eq!(23_962, solve_id("instancesWith5items/423"));
    }
    #[test]
    fn instance_with_5_items_424() {
        assert_eq!(18_159, solve_id("instancesWith5items/424"));
    }
    #[test]
    fn instance_with_5_items_425() {
        assert_eq!(18_452, solve_id("instancesWith5items/425"));
    }
    #[test]
    fn instance_with_5_items_426() {
        assert_eq!(20_269, solve_id("instancesWith5items/426"));
    }
    #[test]
    fn instance_with_5_items_427() {
        assert_eq!(20_812, solve_id("instancesWith5items/427"));
    }
    #[test]
    fn instance_with_5_items_428() {
        assert_eq!(20_372, solve_id("instancesWith5items/428"));
    }
    #[test]
    fn instance_with_5_items_429() {
        assert_eq!(19_641, solve_id("instancesWith5items/429"));
    }
    #[test]
    fn instance_with_5_items_430() {
        assert_eq!(18_455, solve_id("instancesWith5items/430"));
    }
    #[test]
    fn instance_with_5_items_431() {
        assert_eq!(23_951, solve_id("instancesWith5items/431"));
    }
    #[test]
    fn instance_with_5_items_432() {
        assert_eq!(16_600, solve_id("instancesWith5items/432"));
    }
    #[test]
    fn instance_with_5_items_433() {
        assert_eq!(18_283, solve_id("instancesWith5items/433"));
    }
    #[test]
    fn instance_with_5_items_434() {
        assert_eq!(17_631, solve_id("instancesWith5items/434"));
    }
    #[test]
    fn instance_with_5_items_435() {
        assert_eq!(18_441, solve_id("instancesWith5items/435"));
    }
    #[test]
    fn instance_with_5_items_436() {
        assert_eq!(17_503, solve_id("instancesWith5items/436"));
    }
    #[test]
    fn instance_with_5_items_437() {
        assert_eq!(24_574, solve_id("instancesWith5items/437"));
    }
    #[test]
    fn instance_with_5_items_438() {
        assert_eq!(25_299, solve_id("instancesWith5items/438"));
    }
    #[test]
    fn instance_with_5_items_439() {
        assert_eq!(21_709, solve_id("instancesWith5items/439"));
    }
    #[test]
    fn instance_with_5_items_440() {
        assert_eq!(24_939, solve_id("instancesWith5items/440"));
    }
    #[test]
    fn instance_with_5_items_441() {
        assert_eq!(18_740, solve_id("instancesWith5items/441"));
    }
    #[test]
    fn instance_with_5_items_442() {
        assert_eq!(30_110, solve_id("instancesWith5items/442"));
    }
    #[test]
    fn instance_with_5_items_443() {
        assert_eq!(17_932, solve_id("instancesWith5items/443"));
    }
    #[test]
    fn instance_with_5_items_444() {
        assert_eq!(25_779, solve_id("instancesWith5items/444"));
    }
    #[test]
    fn instance_with_5_items_445() {
        assert_eq!(19_029, solve_id("instancesWith5items/445"));
    }
    #[test]
    fn instance_with_5_items_446() {
        assert_eq!(24_179, solve_id("instancesWith5items/446"));
    }
    #[test]
    fn instance_with_5_items_447() {
        assert_eq!(18_062, solve_id("instancesWith5items/447"));
    }
    #[test]
    fn instance_with_5_items_448() {
        assert_eq!(19_557, solve_id("instancesWith5items/448"));
    }
    #[test]
    fn instance_with_5_items_449() {
        assert_eq!(20_259, solve_id("instancesWith5items/449"));
    }
    #[test]
    fn instance_with_5_items_450() {
        assert_eq!(22_151, solve_id("instancesWith5items/450"));
    }
    #[test]
    fn instance_with_5_items_451() {
        assert_eq!(19_592, solve_id("instancesWith5items/451"));
    }
    #[test]
    fn instance_with_5_items_452() {
        assert_eq!(15_798, solve_id("instancesWith5items/452"));
    }
    #[test]
    fn instance_with_5_items_453() {
        assert_eq!(24_977, solve_id("instancesWith5items/453"));
    }
    #[test]
    fn instance_with_5_items_454() {
        assert_eq!(19_405, solve_id("instancesWith5items/454"));
    }
    #[test]
    fn instance_with_5_items_455() {
        assert_eq!(21_167, solve_id("instancesWith5items/455"));
    }
    #[test]
    fn instance_with_5_items_456() {
        assert_eq!(26_882, solve_id("instancesWith5items/456"));
    }
    #[test]
    fn instance_with_5_items_457() {
        assert_eq!(25_581, solve_id("instancesWith5items/457"));
    }
    #[test]
    fn instance_with_5_items_458() {
        assert_eq!(16_724, solve_id("instancesWith5items/458"));
    }
    #[test]
    fn instance_with_5_items_459() {
        assert_eq!(24_433, solve_id("instancesWith5items/459"));
    }
    #[test]
    fn instance_with_5_items_460() {
        assert_eq!(26_595, solve_id("instancesWith5items/460"));
    }
    #[test]
    fn instance_with_5_items_461() {
        assert_eq!(21_192, solve_id("instancesWith5items/461"));
    }
    #[test]
    fn instance_with_5_items_462() {
        assert_eq!(26_619, solve_id("instancesWith5items/462"));
    }
    #[test]
    fn instance_with_5_items_463() {
        assert_eq!(22_770, solve_id("instancesWith5items/463"));
    }
    #[test]
    fn instance_with_5_items_464() {
        assert_eq!(14_739, solve_id("instancesWith5items/464"));
    }
    #[test]
    fn instance_with_5_items_465() {
        assert_eq!(19_996, solve_id("instancesWith5items/465"));
    }
    #[test]
    fn instance_with_5_items_466() {
        assert_eq!(18_391, solve_id("instancesWith5items/466"));
    }
    #[test]
    fn instance_with_5_items_467() {
        assert_eq!(21_723, solve_id("instancesWith5items/467"));
    }
    #[test]
    fn instance_with_5_items_468() {
        assert_eq!(23_313, solve_id("instancesWith5items/468"));
    }
    #[test]
    fn instance_with_5_items_469() {
        assert_eq!(18_181, solve_id("instancesWith5items/469"));
    }
    #[test]
    fn instance_with_5_items_470() {
        assert_eq!(21_876, solve_id("instancesWith5items/470"));
    }
    #[test]
    fn instance_with_5_items_471() {
        assert_eq!(23_490, solve_id("instancesWith5items/471"));
    }
    #[test]
    fn instance_with_5_items_472() {
        assert_eq!(19_718, solve_id("instancesWith5items/472"));
    }
    #[test]
    fn instance_with_5_items_473() {
        assert_eq!(21_943, solve_id("instancesWith5items/473"));
    }
    #[test]
    fn instance_with_5_items_474() {
        assert_eq!(21_982, solve_id("instancesWith5items/474"));
    }
    #[test]
    fn instance_with_5_items_475() {
        assert_eq!(17_546, solve_id("instancesWith5items/475"));
    }
    #[test]
    fn instance_with_5_items_476() {
        assert_eq!(22_531, solve_id("instancesWith5items/476"));
    }
    #[test]
    fn instance_with_5_items_477() {
        assert_eq!(18_847, solve_id("instancesWith5items/477"));
    }
    #[test]
    fn instance_with_5_items_478() {
        assert_eq!(20_327, solve_id("instancesWith5items/478"));
    }
    #[test]
    fn instance_with_5_items_479() {
        assert_eq!(22_812, solve_id("instancesWith5items/479"));
    }
    #[test]
    fn instance_with_5_items_480() {
        assert_eq!(22_144, solve_id("instancesWith5items/480"));
    }
    #[test]
    fn instance_with_5_items_481() {
        assert_eq!(17_083, solve_id("instancesWith5items/481"));
    }
    #[test]
    fn instance_with_5_items_482() {
        assert_eq!(20_730, solve_id("instancesWith5items/482"));
    }
    #[test]
    fn instance_with_5_items_483() {
        assert_eq!(23_758, solve_id("instancesWith5items/483"));
    }
    #[test]
    fn instance_with_5_items_484() {
        assert_eq!(25_415, solve_id("instancesWith5items/484"));
    }
    #[test]
    fn instance_with_5_items_485() {
        assert_eq!(18_365, solve_id("instancesWith5items/485"));
    }
    #[test]
    fn instance_with_5_items_486() {
        assert_eq!(25_352, solve_id("instancesWith5items/486"));
    }
    #[test]
    fn instance_with_5_items_487() {
        assert_eq!(12_560, solve_id("instancesWith5items/487"));
    }
    #[test]
    fn instance_with_5_items_488() {
        assert_eq!(25_007, solve_id("instancesWith5items/488"));
    }
    #[test]
    fn instance_with_5_items_489() {
        assert_eq!(22_577, solve_id("instancesWith5items/489"));
    }
    #[test]
    fn instance_with_5_items_490() {
        assert_eq!(21_859, solve_id("instancesWith5items/490"));
    }
    #[test]
    fn instance_with_5_items_491() {
        assert_eq!(20_287, solve_id("instancesWith5items/491"));
    }
    #[test]
    fn instance_with_5_items_492() {
        assert_eq!(17_282, solve_id("instancesWith5items/492"));
    }
    #[test]
    fn instance_with_5_items_493() {
        assert_eq!(22_292, solve_id("instancesWith5items/493"));
    }
    #[test]
    fn instance_with_5_items_494() {
        assert_eq!(23_053, solve_id("instancesWith5items/494"));
    }
    #[test]
    fn instance_with_5_items_495() {
        assert_eq!(29_366, solve_id("instancesWith5items/495"));
    }
    #[test]
    fn instance_with_5_items_496() {
        assert_eq!(28_399, solve_id("instancesWith5items/496"));
    }
    #[test]
    fn instance_with_5_items_497() {
        assert_eq!(16_189, solve_id("instancesWith5items/497"));
    }
    #[test]
    fn instance_with_5_items_498() {
        assert_eq!(21_431, solve_id("instancesWith5items/498"));
    }
    #[test]
    fn instance_with_5_items_499() {
        assert_eq!(18_968, solve_id("instancesWith5items/499"));
    }
    #[test]
    fn instance_with_5_items_500() {
        assert_eq!(24_676, solve_id("instancesWith5items/500"));
    }
    #[test]
    fn instance_with_5_items_501() {
        assert_eq!(22_281, solve_id("instancesWith5items/501"));
    }
    #[test]
    fn instance_with_5_items_502() {
        assert_eq!(72_174, solve_id("instancesWith5items/502"));
    }
    #[test]
    fn instance_with_5_items_503() {
        assert_eq!(63_544, solve_id("instancesWith5items/503"));
    }
    #[test]
    fn instance_with_5_items_504() {
        assert_eq!(23_713, solve_id("instancesWith5items/504"));
    }
    #[test]
    fn instance_with_5_items_505() {
        assert_eq!(37_266, solve_id("instancesWith5items/505"));
    }
    #[test]
    fn instance_with_5_items_506() {
        assert_eq!(63_237, solve_id("instancesWith5items/506"));
    }
    #[test]
    fn instance_with_5_items_507() {
        assert_eq!(29_146, solve_id("instancesWith5items/507"));
    }
    #[test]
    fn instance_with_5_items_508() {
        assert_eq!(46_533, solve_id("instancesWith5items/508"));
    }
    #[test]
    fn instance_with_5_items_509() {
        assert_eq!(89_761, solve_id("instancesWith5items/509"));
    }
    #[test]
    fn instance_with_5_items_510() {
        assert_eq!(62_323, solve_id("instancesWith5items/510"));
    }
    #[test]
    fn instance_with_5_items_511() {
        assert_eq!(34_350, solve_id("instancesWith5items/511"));
    }
    #[test]
    fn instance_with_5_items_512() {
        assert_eq!(56_500, solve_id("instancesWith5items/512"));
    }
    #[test]
    fn instance_with_5_items_513() {
        assert_eq!(29_220, solve_id("instancesWith5items/513"));
    }
    #[test]
    fn instance_with_5_items_514() {
        assert_eq!(46_455, solve_id("instancesWith5items/514"));
    }
    #[test]
    fn instance_with_5_items_515() {
        assert_eq!(75_046, solve_id("instancesWith5items/515"));
    }
    #[test]
    fn instance_with_5_items_516() {
        assert_eq!(51_505, solve_id("instancesWith5items/516"));
    }
    #[test]
    fn instance_with_5_items_517() {
        assert_eq!(38_914, solve_id("instancesWith5items/517"));
    }
    #[test]
    fn instance_with_5_items_518() {
        assert_eq!(86_901, solve_id("instancesWith5items/518"));
    }
    #[test]
    fn instance_with_5_items_519() {
        assert_eq!(45_741, solve_id("instancesWith5items/519"));
    }
    #[test]
    fn instance_with_5_items_520() {
        assert_eq!(50_899, solve_id("instancesWith5items/520"));
    }
    #[test]
    fn instance_with_5_items_521() {
        assert_eq!(24_768, solve_id("instancesWith5items/521"));
    }
    #[test]
    fn instance_with_5_items_522() {
        assert_eq!(49_104, solve_id("instancesWith5items/522"));
    }
    #[test]
    fn instance_with_5_items_523() {
        assert_eq!(40_276, solve_id("instancesWith5items/523"));
    }
    #[test]
    fn instance_with_5_items_524() {
        assert_eq!(46_471, solve_id("instancesWith5items/524"));
    }
    #[test]
    fn instance_with_5_items_525() {
        assert_eq!(51_714, solve_id("instancesWith5items/525"));
    }
    #[test]
    fn instance_with_5_items_526() {
        assert_eq!(51_434, solve_id("instancesWith5items/526"));
    }
    #[test]
    fn instance_with_5_items_527() {
        assert_eq!(55_854, solve_id("instancesWith5items/527"));
    }
    #[test]
    fn instance_with_5_items_528() {
        assert_eq!(63_002, solve_id("instancesWith5items/528"));
    }
    #[test]
    fn instance_with_5_items_529() {
        assert_eq!(38_998, solve_id("instancesWith5items/529"));
    }
    #[test]
    fn instance_with_5_items_530() {
        assert_eq!(55_118, solve_id("instancesWith5items/530"));
    }
    #[test]
    fn instance_with_5_items_531() {
        assert_eq!(65_283, solve_id("instancesWith5items/531"));
    }
    #[test]
    fn instance_with_5_items_532() {
        assert_eq!(113_784, solve_id("instancesWith5items/532"));
    }
    #[test]
    fn instance_with_5_items_533() {
        assert_eq!(36_923, solve_id("instancesWith5items/533"));
    }
    #[test]
    fn instance_with_5_items_534() {
        assert_eq!(38_261, solve_id("instancesWith5items/534"));
    }
    #[test]
    fn instance_with_5_items_535() {
        assert_eq!(51_077, solve_id("instancesWith5items/535"));
    }
    #[test]
    fn instance_with_5_items_536() {
        assert_eq!(40_318, solve_id("instancesWith5items/536"));
    }
    #[test]
    fn instance_with_5_items_537() {
        assert_eq!(74_751, solve_id("instancesWith5items/537"));
    }
    #[test]
    fn instance_with_5_items_538() {
        assert_eq!(51_203, solve_id("instancesWith5items/538"));
    }
    #[test]
    fn instance_with_5_items_539() {
        assert_eq!(57_465, solve_id("instancesWith5items/539"));
    }
    #[test]
    fn instance_with_5_items_540() {
        assert_eq!(43_833, solve_id("instancesWith5items/540"));
    }
    #[test]
    fn instance_with_5_items_541() {
        assert_eq!(67_123, solve_id("instancesWith5items/541"));
    }
    #[test]
    fn instance_with_5_items_542() {
        assert_eq!(32_843, solve_id("instancesWith5items/542"));
    }
    #[test]
    fn instance_with_5_items_543() {
        assert_eq!(88_455, solve_id("instancesWith5items/543"));
    }
    #[test]
    fn instance_with_5_items_544() {
        assert_eq!(44_757, solve_id("instancesWith5items/544"));
    }
    #[test]
    fn instance_with_5_items_545() {
        assert_eq!(33_270, solve_id("instancesWith5items/545"));
    }
    #[test]
    fn instance_with_5_items_546() {
        assert_eq!(128_740, solve_id("instancesWith5items/546"));
    }
    #[test]
    fn instance_with_5_items_547() {
        assert_eq!(41_321, solve_id("instancesWith5items/547"));
    }
    #[test]
    fn instance_with_5_items_548() {
        assert_eq!(28_324, solve_id("instancesWith5items/548"));
    }
    #[test]
    fn instance_with_5_items_549() {
        assert_eq!(28_096, solve_id("instancesWith5items/549"));
    }
    #[test]
    fn instance_with_5_items_550() {
        assert_eq!(77_777, solve_id("instancesWith5items/550"));
    }
    #[test]
    fn instance_with_5_items_551() {
        assert_eq!(34_800, solve_id("instancesWith5items/551"));
    }
    #[test]
    fn instance_with_5_items_552() {
        assert_eq!(53_439, solve_id("instancesWith5items/552"));
    }
    #[test]
    fn instance_with_5_items_553() {
        assert_eq!(50_540, solve_id("instancesWith5items/553"));
    }
    #[test]
    fn instance_with_5_items_554() {
        assert_eq!(48_963, solve_id("instancesWith5items/554"));
    }
    #[test]
    fn instance_with_5_items_555() {
        assert_eq!(31_164, solve_id("instancesWith5items/555"));
    }
    #[test]
    fn instance_with_5_items_556() {
        assert_eq!(65_633, solve_id("instancesWith5items/556"));
    }
    #[test]
    fn instance_with_5_items_557() {
        assert_eq!(55_668, solve_id("instancesWith5items/557"));
    }
    #[test]
    fn instance_with_5_items_558() {
        assert_eq!(86_042, solve_id("instancesWith5items/558"));
    }
    #[test]
    fn instance_with_5_items_559() {
        assert_eq!(63_569, solve_id("instancesWith5items/559"));
    }
    #[test]
    fn instance_with_5_items_560() {
        assert_eq!(56_445, solve_id("instancesWith5items/560"));
    }
    #[test]
    fn instance_with_5_items_561() {
        assert_eq!(55_688, solve_id("instancesWith5items/561"));
    }
    #[test]
    fn instance_with_5_items_562() {
        assert_eq!(36_889, solve_id("instancesWith5items/562"));
    }
    #[test]
    fn instance_with_5_items_563() {
        assert_eq!(68_344, solve_id("instancesWith5items/563"));
    }
    #[test]
    fn instance_with_5_items_564() {
        assert_eq!(44_757, solve_id("instancesWith5items/564"));
    }
    #[test]
    fn instance_with_5_items_565() {
        assert_eq!(31_251, solve_id("instancesWith5items/565"));
    }
    #[test]
    fn instance_with_5_items_566() {
        assert_eq!(42_851, solve_id("instancesWith5items/566"));
    }
    #[test]
    fn instance_with_5_items_567() {
        assert_eq!(40_480, solve_id("instancesWith5items/567"));
    }
    #[test]
    fn instance_with_5_items_568() {
        assert_eq!(42_097, solve_id("instancesWith5items/568"));
    }
    #[test]
    fn instance_with_5_items_569() {
        assert_eq!(68_904, solve_id("instancesWith5items/569"));
    }
    #[test]
    fn instance_with_5_items_570() {
        assert_eq!(40_464, solve_id("instancesWith5items/570"));
    }
    #[test]
    fn instance_with_5_items_571() {
        assert_eq!(46_216, solve_id("instancesWith5items/571"));
    }
    #[test]
    fn instance_with_5_items_572() {
        assert_eq!(105_474, solve_id("instancesWith5items/572"));
    }
    #[test]
    fn instance_with_5_items_573() {
        assert_eq!(108_595, solve_id("instancesWith5items/573"));
    }
    #[test]
    fn instance_with_5_items_574() {
        assert_eq!(44_916, solve_id("instancesWith5items/574"));
    }
    #[test]
    fn instance_with_5_items_575() {
        assert_eq!(43_889, solve_id("instancesWith5items/575"));
    }
    #[test]
    fn instance_with_5_items_576() {
        assert_eq!(43_986, solve_id("instancesWith5items/576"));
    }
    #[test]
    fn instance_with_5_items_577() {
        assert_eq!(51_039, solve_id("instancesWith5items/577"));
    }
    #[test]
    fn instance_with_5_items_578() {
        assert_eq!(43_119, solve_id("instancesWith5items/578"));
    }
    #[test]
    fn instance_with_5_items_579() {
        assert_eq!(60_241, solve_id("instancesWith5items/579"));
    }
    #[test]
    fn instance_with_5_items_580() {
        assert_eq!(52_132, solve_id("instancesWith5items/580"));
    }
    #[test]
    fn instance_with_5_items_581() {
        assert_eq!(42_860, solve_id("instancesWith5items/581"));
    }
    #[test]
    fn instance_with_5_items_582() {
        assert_eq!(28_752, solve_id("instancesWith5items/582"));
    }
    #[test]
    fn instance_with_5_items_583() {
        assert_eq!(54_910, solve_id("instancesWith5items/583"));
    }
    #[test]
    fn instance_with_5_items_584() {
        assert_eq!(55_287, solve_id("instancesWith5items/584"));
    }
    #[test]
    fn instance_with_5_items_585() {
        assert_eq!(38_784, solve_id("instancesWith5items/585"));
    }
    #[test]
    fn instance_with_5_items_586() {
        assert_eq!(29_757, solve_id("instancesWith5items/586"));
    }
    #[test]
    fn instance_with_5_items_587() {
        assert_eq!(78_619, solve_id("instancesWith5items/587"));
    }
    #[test]
    fn instance_with_5_items_588() {
        assert_eq!(70_858, solve_id("instancesWith5items/588"));
    }
    #[test]
    fn instance_with_5_items_589() {
        assert_eq!(86_787, solve_id("instancesWith5items/589"));
    }
    #[test]
    fn instance_with_5_items_590() {
        assert_eq!(40_577, solve_id("instancesWith5items/590"));
    }
    #[test]
    fn instance_with_5_items_591() {
        assert_eq!(39_033, solve_id("instancesWith5items/591"));
    }
    #[test]
    fn instance_with_5_items_592() {
        assert_eq!(56_147, solve_id("instancesWith5items/592"));
    }
    #[test]
    fn instance_with_5_items_593() {
        assert_eq!(48_934, solve_id("instancesWith5items/593"));
    }
    #[test]
    fn instance_with_5_items_594() {
        assert_eq!(95_315, solve_id("instancesWith5items/594"));
    }
    #[test]
    fn instance_with_5_items_595() {
        assert_eq!(91_065, solve_id("instancesWith5items/595"));
    }
    #[test]
    fn instance_with_5_items_596() {
        assert_eq!(48_625, solve_id("instancesWith5items/596"));
    }
    #[test]
    fn instance_with_5_items_597() {
        assert_eq!(34_335, solve_id("instancesWith5items/597"));
    }
    #[test]
    fn instance_with_5_items_598() {
        assert_eq!(68_479, solve_id("instancesWith5items/598"));
    }
    #[test]
    fn instance_with_5_items_599() {
        assert_eq!(46_246, solve_id("instancesWith5items/599"));
    }
    #[test]
    fn instance_with_5_items_600() {
        assert_eq!(47_697, solve_id("instancesWith5items/600"));
    }
    #[test]
    fn instance_with_5_items_601() {
        assert_eq!(62_821, solve_id("instancesWith5items/601"));
    }
    #[test]
    fn instance_with_5_items_602() {
        assert_eq!(56_000, solve_id("instancesWith5items/602"));
    }
    #[test]
    fn instance_with_5_items_603() {
        assert_eq!(26_124, solve_id("instancesWith5items/603"));
    }
    #[test]
    fn instance_with_5_items_604() {
        assert_eq!(37_076, solve_id("instancesWith5items/604"));
    }
    #[test]
    fn instance_with_5_items_605() {
        assert_eq!(30_494, solve_id("instancesWith5items/605"));
    }
    #[test]
    fn instance_with_5_items_606() {
        assert_eq!(28_224, solve_id("instancesWith5items/606"));
    }
    #[test]
    fn instance_with_5_items_607() {
        assert_eq!(38_317, solve_id("instancesWith5items/607"));
    }
    #[test]
    fn instance_with_5_items_608() {
        assert_eq!(60_279, solve_id("instancesWith5items/608"));
    }
    #[test]
    fn instance_with_5_items_609() {
        assert_eq!(44_916, solve_id("instancesWith5items/609"));
    }
    #[test]
    fn instance_with_5_items_610() {
        assert_eq!(44_813, solve_id("instancesWith5items/610"));
    }
    #[test]
    fn instance_with_5_items_611() {
        assert_eq!(38_281, solve_id("instancesWith5items/611"));
    }
    #[test]
    fn instance_with_5_items_612() {
        assert_eq!(49_196, solve_id("instancesWith5items/612"));
    }
    #[test]
    fn instance_with_5_items_613() {
        assert_eq!(50_529, solve_id("instancesWith5items/613"));
    }
    #[test]
    fn instance_with_5_items_614() {
        assert_eq!(40_093, solve_id("instancesWith5items/614"));
    }
    #[test]
    fn instance_with_5_items_615() {
        assert_eq!(39_431, solve_id("instancesWith5items/615"));
    }
    #[test]
    fn instance_with_5_items_616() {
        assert_eq!(45_723, solve_id("instancesWith5items/616"));
    }
    #[test]
    fn instance_with_5_items_617() {
        assert_eq!(61_411, solve_id("instancesWith5items/617"));
    }
    #[test]
    fn instance_with_5_items_618() {
        assert_eq!(53_188, solve_id("instancesWith5items/618"));
    }
    #[test]
    fn instance_with_5_items_619() {
        assert_eq!(41_476, solve_id("instancesWith5items/619"));
    }
    #[test]
    fn instance_with_5_items_620() {
        assert_eq!(54_418, solve_id("instancesWith5items/620"));
    }
    #[test]
    fn instance_with_5_items_621() {
        assert_eq!(39_857, solve_id("instancesWith5items/621"));
    }
    #[test]
    fn instance_with_5_items_622() {
        assert_eq!(45_527, solve_id("instancesWith5items/622"));
    }
    #[test]
    fn instance_with_5_items_623() {
        assert_eq!(41_456, solve_id("instancesWith5items/623"));
    }
    #[test]
    fn instance_with_5_items_624() {
        assert_eq!(48_960, solve_id("instancesWith5items/624"));
    }
    #[test]
    fn instance_with_5_items_625() {
        assert_eq!(43_947, solve_id("instancesWith5items/625"));
    }
    #[test]
    fn instance_with_5_items_626() {
        assert_eq!(48_882, solve_id("instancesWith5items/626"));
    }
    #[test]
    fn instance_with_5_items_627() {
        assert_eq!(35_988, solve_id("instancesWith5items/627"));
    }
    #[test]
    fn instance_with_5_items_628() {
        assert_eq!(39_260, solve_id("instancesWith5items/628"));
    }
    #[test]
    fn instance_with_5_items_629() {
        assert_eq!(55_265, solve_id("instancesWith5items/629"));
    }
    #[test]
    fn instance_with_5_items_630() {
        assert_eq!(55_763, solve_id("instancesWith5items/630"));
    }
    #[test]
    fn instance_with_5_items_631() {
        assert_eq!(48_909, solve_id("instancesWith5items/631"));
    }
    #[test]
    fn instance_with_5_items_632() {
        assert_eq!(43_418, solve_id("instancesWith5items/632"));
    }
    #[test]
    fn instance_with_5_items_633() {
        assert_eq!(43_639, solve_id("instancesWith5items/633"));
    }
    #[test]
    fn instance_with_5_items_634() {
        assert_eq!(40_024, solve_id("instancesWith5items/634"));
    }
    #[test]
    fn instance_with_5_items_635() {
        assert_eq!(50_612, solve_id("instancesWith5items/635"));
    }
    #[test]
    fn instance_with_5_items_636() {
        assert_eq!(45_718, solve_id("instancesWith5items/636"));
    }
    #[test]
    fn instance_with_5_items_637() {
        assert_eq!(55_336, solve_id("instancesWith5items/637"));
    }
    #[test]
    fn instance_with_5_items_638() {
        assert_eq!(64_626, solve_id("instancesWith5items/638"));
    }
    #[test]
    fn instance_with_5_items_639() {
        assert_eq!(45_054, solve_id("instancesWith5items/639"));
    }
    #[test]
    fn instance_with_5_items_640() {
        assert_eq!(45_216, solve_id("instancesWith5items/640"));
    }
    #[test]
    fn instance_with_5_items_641() {
        assert_eq!(47_927, solve_id("instancesWith5items/641"));
    }
    #[test]
    fn instance_with_5_items_642() {
        assert_eq!(36_694, solve_id("instancesWith5items/642"));
    }
    #[test]
    fn instance_with_5_items_643() {
        assert_eq!(44_876, solve_id("instancesWith5items/643"));
    }
    #[test]
    fn instance_with_5_items_644() {
        assert_eq!(49_910, solve_id("instancesWith5items/644"));
    }
    #[test]
    fn instance_with_5_items_645() {
        assert_eq!(44_987, solve_id("instancesWith5items/645"));
    }
    #[test]
    fn instance_with_5_items_646() {
        assert_eq!(48_263, solve_id("instancesWith5items/646"));
    }
    #[test]
    fn instance_with_5_items_647() {
        assert_eq!(37_813, solve_id("instancesWith5items/647"));
    }
    #[test]
    fn instance_with_5_items_648() {
        assert_eq!(46_021, solve_id("instancesWith5items/648"));
    }
    #[test]
    fn instance_with_5_items_649() {
        assert_eq!(41_985, solve_id("instancesWith5items/649"));
    }
    #[test]
    fn instance_with_5_items_650() {
        assert_eq!(48_004, solve_id("instancesWith5items/650"));
    }
    #[test]
    fn instance_with_5_items_651() {
        assert_eq!(32_576, solve_id("instancesWith5items/651"));
    }
    #[test]
    fn instance_with_5_items_652() {
        assert_eq!(47_275, solve_id("instancesWith5items/652"));
    }
    #[test]
    fn instance_with_5_items_653() {
        assert_eq!(40_232, solve_id("instancesWith5items/653"));
    }
    #[test]
    fn instance_with_5_items_654() {
        assert_eq!(21_953, solve_id("instancesWith5items/654"));
    }
    #[test]
    fn instance_with_5_items_655() {
        assert_eq!(45_625, solve_id("instancesWith5items/655"));
    }
    #[test]
    fn instance_with_5_items_656() {
        assert_eq!(45_023, solve_id("instancesWith5items/656"));
    }
    #[test]
    fn instance_with_5_items_657() {
        assert_eq!(51_558, solve_id("instancesWith5items/657"));
    }
    #[test]
    fn instance_with_5_items_658() {
        assert_eq!(55_652, solve_id("instancesWith5items/658"));
    }
    #[test]
    fn instance_with_5_items_659() {
        assert_eq!(43_336, solve_id("instancesWith5items/659"));
    }
    #[test]
    fn instance_with_5_items_660() {
        assert_eq!(45_852, solve_id("instancesWith5items/660"));
    }
    #[test]
    fn instance_with_5_items_661() {
        assert_eq!(41_493, solve_id("instancesWith5items/661"));
    }
    #[test]
    fn instance_with_5_items_662() {
        assert_eq!(48_150, solve_id("instancesWith5items/662"));
    }
    #[test]
    fn instance_with_5_items_663() {
        assert_eq!(43_313, solve_id("instancesWith5items/663"));
    }
    #[test]
    fn instance_with_5_items_664() {
        assert_eq!(47_176, solve_id("instancesWith5items/664"));
    }
    #[test]
    fn instance_with_5_items_665() {
        assert_eq!(41_637, solve_id("instancesWith5items/665"));
    }
    #[test]
    fn instance_with_5_items_666() {
        assert_eq!(46_649, solve_id("instancesWith5items/666"));
    }
    #[test]
    fn instance_with_5_items_667() {
        assert_eq!(55_928, solve_id("instancesWith5items/667"));
    }
    #[test]
    fn instance_with_5_items_668() {
        assert_eq!(41_296, solve_id("instancesWith5items/668"));
    }
    #[test]
    fn instance_with_5_items_669() {
        assert_eq!(42_123, solve_id("instancesWith5items/669"));
    }
    #[test]
    fn instance_with_5_items_670() {
        assert_eq!(36_100, solve_id("instancesWith5items/670"));
    }
    #[test]
    fn instance_with_5_items_671() {
        assert_eq!(55_049, solve_id("instancesWith5items/671"));
    }
    #[test]
    fn instance_with_5_items_672() {
        assert_eq!(43_385, solve_id("instancesWith5items/672"));
    }
    #[test]
    fn instance_with_5_items_673() {
        assert_eq!(46_053, solve_id("instancesWith5items/673"));
    }
    #[test]
    fn instance_with_5_items_674() {
        assert_eq!(45_320, solve_id("instancesWith5items/674"));
    }
    #[test]
    fn instance_with_5_items_675() {
        assert_eq!(61_843, solve_id("instancesWith5items/675"));
    }
    #[test]
    fn instance_with_5_items_676() {
        assert_eq!(34_255, solve_id("instancesWith5items/676"));
    }
    #[test]
    fn instance_with_5_items_677() {
        assert_eq!(40_747, solve_id("instancesWith5items/677"));
    }
    #[test]
    fn instance_with_5_items_678() {
        assert_eq!(43_925, solve_id("instancesWith5items/678"));
    }
    #[test]
    fn instance_with_5_items_679() {
        assert_eq!(49_901, solve_id("instancesWith5items/679"));
    }
    #[test]
    fn instance_with_5_items_680() {
        assert_eq!(42_696, solve_id("instancesWith5items/680"));
    }
    #[test]
    fn instance_with_5_items_681() {
        assert_eq!(46_380, solve_id("instancesWith5items/681"));
    }
    #[test]
    fn instance_with_5_items_682() {
        assert_eq!(43_277, solve_id("instancesWith5items/682"));
    }
    #[test]
    fn instance_with_5_items_683() {
        assert_eq!(48_165, solve_id("instancesWith5items/683"));
    }
    #[test]
    fn instance_with_5_items_684() {
        assert_eq!(40_141, solve_id("instancesWith5items/684"));
    }
    #[test]
    fn instance_with_5_items_685() {
        assert_eq!(49_011, solve_id("instancesWith5items/685"));
    }
    #[test]
    fn instance_with_5_items_686() {
        assert_eq!(41_405, solve_id("instancesWith5items/686"));
    }
    #[test]
    fn instance_with_5_items_687() {
        assert_eq!(45_510, solve_id("instancesWith5items/687"));
    }
    #[test]
    fn instance_with_5_items_688() {
        assert_eq!(32_776, solve_id("instancesWith5items/688"));
    }
    #[test]
    fn instance_with_5_items_689() {
        assert_eq!(48_928, solve_id("instancesWith5items/689"));
    }
    #[test]
    fn instance_with_5_items_690() {
        assert_eq!(63_170, solve_id("instancesWith5items/690"));
    }
    #[test]
    fn instance_with_5_items_691() {
        assert_eq!(48_089, solve_id("instancesWith5items/691"));
    }
    #[test]
    fn instance_with_5_items_692() {
        assert_eq!(45_756, solve_id("instancesWith5items/692"));
    }
    #[test]
    fn instance_with_5_items_693() {
        assert_eq!(36_442, solve_id("instancesWith5items/693"));
    }
    #[test]
    fn instance_with_5_items_694() {
        assert_eq!(57_314, solve_id("instancesWith5items/694"));
    }
    #[test]
    fn instance_with_5_items_695() {
        assert_eq!(41_397, solve_id("instancesWith5items/695"));
    }
    #[test]
    fn instance_with_5_items_696() {
        assert_eq!(42_224, solve_id("instancesWith5items/696"));
    }
    #[test]
    fn instance_with_5_items_697() {
        assert_eq!(38_830, solve_id("instancesWith5items/697"));
    }
    #[test]
    fn instance_with_5_items_698() {
        assert_eq!(40_382, solve_id("instancesWith5items/698"));
    }
    #[test]
    fn instance_with_5_items_699() {
        assert_eq!(40_560, solve_id("instancesWith5items/699"));
    }
    #[test]
    fn instance_with_5_items_700() {
        assert_eq!(32_679, solve_id("instancesWith5items/700"));
    }
    #[test]
    fn instance_with_5_items_701() {
        assert_eq!(65_170, solve_id("instancesWith5items/701"));
    }
    #[test]
    fn instance_with_5_items_702() {
        assert_eq!(58_763, solve_id("instancesWith5items/702"));
    }
    #[test]
    fn instance_with_5_items_703() {
        assert_eq!(29_013, solve_id("instancesWith5items/703"));
    }
    #[test]
    fn instance_with_5_items_704() {
        assert_eq!(40_043, solve_id("instancesWith5items/704"));
    }
    #[test]
    fn instance_with_5_items_705() {
        assert_eq!(32_806, solve_id("instancesWith5items/705"));
    }
    #[test]
    fn instance_with_5_items_706() {
        assert_eq!(31_173, solve_id("instancesWith5items/706"));
    }
    #[test]
    fn instance_with_5_items_707() {
        assert_eq!(41_028, solve_id("instancesWith5items/707"));
    }
    #[test]
    fn instance_with_5_items_708() {
        assert_eq!(62_846, solve_id("instancesWith5items/708"));
    }
    #[test]
    fn instance_with_5_items_709() {
        assert_eq!(47_503, solve_id("instancesWith5items/709"));
    }
    #[test]
    fn instance_with_5_items_710() {
        assert_eq!(46_082, solve_id("instancesWith5items/710"));
    }
    #[test]
    fn instance_with_5_items_711() {
        assert_eq!(40_846, solve_id("instancesWith5items/711"));
    }
    #[test]
    fn instance_with_5_items_712() {
        assert_eq!(51_524, solve_id("instancesWith5items/712"));
    }
    #[test]
    fn instance_with_5_items_713() {
        assert_eq!(52_963, solve_id("instancesWith5items/713"));
    }
    #[test]
    fn instance_with_5_items_714() {
        assert_eq!(41_856, solve_id("instancesWith5items/714"));
    }
    #[test]
    fn instance_with_5_items_715() {
        assert_eq!(41_694, solve_id("instancesWith5items/715"));
    }
    #[test]
    fn instance_with_5_items_716() {
        assert_eq!(48_191, solve_id("instancesWith5items/716"));
    }
    #[test]
    fn instance_with_5_items_717() {
        assert_eq!(63_371, solve_id("instancesWith5items/717"));
    }
    #[test]
    fn instance_with_5_items_718() {
        assert_eq!(55_206, solve_id("instancesWith5items/718"));
    }
    #[test]
    fn instance_with_5_items_719() {
        assert_eq!(44_277, solve_id("instancesWith5items/719"));
    }
    #[test]
    fn instance_with_5_items_720() {
        assert_eq!(56_321, solve_id("instancesWith5items/720"));
    }
    #[test]
    fn instance_with_5_items_721() {
        assert_eq!(42_329, solve_id("instancesWith5items/721"));
    }
    #[test]
    fn instance_with_5_items_722() {
        assert_eq!(48_269, solve_id("instancesWith5items/722"));
    }
    #[test]
    fn instance_with_5_items_723() {
        assert_eq!(43_803, solve_id("instancesWith5items/723"));
    }
    #[test]
    fn instance_with_5_items_724() {
        assert_eq!(51_083, solve_id("instancesWith5items/724"));
    }
    #[test]
    fn instance_with_5_items_725() {
        assert_eq!(46_775, solve_id("instancesWith5items/725"));
    }
    #[test]
    fn instance_with_5_items_726() {
        assert_eq!(51_687, solve_id("instancesWith5items/726"));
    }
    #[test]
    fn instance_with_5_items_727() {
        assert_eq!(38_839, solve_id("instancesWith5items/727"));
    }
    #[test]
    fn instance_with_5_items_728() {
        assert_eq!(41_755, solve_id("instancesWith5items/728"));
    }
    #[test]
    fn instance_with_5_items_729() {
        assert_eq!(57_421, solve_id("instancesWith5items/729"));
    }
    #[test]
    fn instance_with_5_items_730() {
        assert_eq!(58_311, solve_id("instancesWith5items/730"));
    }
    #[test]
    fn instance_with_5_items_731() {
        assert_eq!(51_077, solve_id("instancesWith5items/731"));
    }
    #[test]
    fn instance_with_5_items_732() {
        assert_eq!(45_691, solve_id("instancesWith5items/732"));
    }
    #[test]
    fn instance_with_5_items_733() {
        assert_eq!(46_482, solve_id("instancesWith5items/733"));
    }
    #[test]
    fn instance_with_5_items_734() {
        assert_eq!(42_999, solve_id("instancesWith5items/734"));
    }
    #[test]
    fn instance_with_5_items_735() {
        assert_eq!(52_591, solve_id("instancesWith5items/735"));
    }
    #[test]
    fn instance_with_5_items_736() {
        assert_eq!(48_092, solve_id("instancesWith5items/736"));
    }
    #[test]
    fn instance_with_5_items_737() {
        assert_eq!(57_549, solve_id("instancesWith5items/737"));
    }
    #[test]
    fn instance_with_5_items_738() {
        assert_eq!(66_792, solve_id("instancesWith5items/738"));
    }
    #[test]
    fn instance_with_5_items_739() {
        assert_eq!(48_049, solve_id("instancesWith5items/739"));
    }
    #[test]
    fn instance_with_5_items_740() {
        assert_eq!(48_115, solve_id("instancesWith5items/740"));
    }
    #[test]
    fn instance_with_5_items_741() {
        assert_eq!(50_735, solve_id("instancesWith5items/741"));
    }
    #[test]
    fn instance_with_5_items_742() {
        assert_eq!(39_165, solve_id("instancesWith5items/742"));
    }
    #[test]
    fn instance_with_5_items_743() {
        assert_eq!(47_627, solve_id("instancesWith5items/743"));
    }
    #[test]
    fn instance_with_5_items_744() {
        assert_eq!(52_307, solve_id("instancesWith5items/744"));
    }
    #[test]
    fn instance_with_5_items_745() {
        assert_eq!(46_984, solve_id("instancesWith5items/745"));
    }
    #[test]
    fn instance_with_5_items_746() {
        assert_eq!(50_710, solve_id("instancesWith5items/746"));
    }
    #[test]
    fn instance_with_5_items_747() {
        assert_eq!(40_854, solve_id("instancesWith5items/747"));
    }
    #[test]
    fn instance_with_5_items_748() {
        assert_eq!(48_712, solve_id("instancesWith5items/748"));
    }
    #[test]
    fn instance_with_5_items_749() {
        assert_eq!(44_511, solve_id("instancesWith5items/749"));
    }
    #[test]
    fn instance_with_5_items_750() {
        assert_eq!(50_312, solve_id("instancesWith5items/750"));
    }
    #[test]
    fn instance_with_5_items_751() {
        assert_eq!(35_321, solve_id("instancesWith5items/751"));
    }
    #[test]
    fn instance_with_5_items_752() {
        assert_eq!(49_934, solve_id("instancesWith5items/752"));
    }
    #[test]
    fn instance_with_5_items_753() {
        assert_eq!(42_907, solve_id("instancesWith5items/753"));
    }
    #[test]
    fn instance_with_5_items_754() {
        assert_eq!(25_257, solve_id("instancesWith5items/754"));
    }
    #[test]
    fn instance_with_5_items_755() {
        assert_eq!(47_916, solve_id("instancesWith5items/755"));
    }
    #[test]
    fn instance_with_5_items_756() {
        assert_eq!(46_481, solve_id("instancesWith5items/756"));
    }
    #[test]
    fn instance_with_5_items_757() {
        assert_eq!(53_926, solve_id("instancesWith5items/757"));
    }
    #[test]
    fn instance_with_5_items_758() {
        assert_eq!(57_455, solve_id("instancesWith5items/758"));
    }
    #[test]
    fn instance_with_5_items_759() {
        assert_eq!(45_707, solve_id("instancesWith5items/759"));
    }
    #[test]
    fn instance_with_5_items_760() {
        assert_eq!(48_334, solve_id("instancesWith5items/760"));
    }
}
*/