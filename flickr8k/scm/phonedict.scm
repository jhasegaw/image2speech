(if (not (member_string "phonedict" (lex.list)))
    (defvar phonedictdir (path-as-directory "/Users/jhasegaw/d/projects/image2speech/flickr8k_cg/"))
    (lex.create "phonedict")
    (lex.set.compile.file (path-append phonedictdir "phonedict.txt"))
    (lex.set.phoneset "radio")))

(define (image2speech_L1phones_flickr8k_cg::select_lexicon)
  "(image2speech_L1phones_flickr8k_cg::select_lexicon)
Set up the CMU lexicon for US English."
  (lex.select "phonedict")

  ;; Post lexical rules
  (set! postlex_rules_hooks nil) 
  (set! postlex_vowel_reduce_cart_tree nil) ; no reduction
)

(define (image2speech_L1phones_flickr8k_cg::reset_lexicon)
  "(image2speech_L1phones_flickr8k_cg::reset_lexicon)
Reset lexicon information."
  t
)

(provide 'image2speech_L1phones_flickr8k_cg_lexicon)


