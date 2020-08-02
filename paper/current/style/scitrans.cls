% $Id: scitrans.cls,v 1.13 2000-12-13 09:17:53+09 aoki Exp $

\ProvidesPackage{scitrans}

\newif\if@english \@englishfalse
\newif\if@twocol  \@twocoltrue
\newif\if@bookrev \@bookrevfalse
\newif\if@tokureverse \@tokureversefalse
\newif\if@includeamsmath \@includeamsmathfalse
\DeclareOption{J}{\@englishfalse}
\DeclareOption{E}{\@englishtrue}
\DeclareOption{S}{\@twocolfalse}
\DeclareOption{B}{\@bookrevtrue} 
\DeclareOption{reverse}{\@tokureversetrue}
\DeclareOption{amsmath}{\@includeamsmathtrue}
\ProcessOptions

\if@tokureverse
  \RequirePackage{color}
\fi

\if@includeamsmath
  \RequirePackage[fleqn]{amsmath}
\fi

\RequirePackage{latexsym}
\@input{ulasy.fd}
\@input{scims.sty}
\def\@magscale#1{ scaled \magstep #1}
\def\@halfmag{ scaled \magstephalf}
\def\@ptscale#1{ scaled #100}

\newdimen\zw
\newdimen\zh
\newdimen\H
\def\@setzw#1{\zw=#1\relax \zh=0.9524296\zw% based on min10.tfm
  \if@english\else
	\kanjiskip 0mm plus.1\zw minus.05\zw
	\xkanjiskip .2\zw plus.2\zw minus.1\zw
  \fi}
\@setzw{3.25mm}% 13Q
\parindent3.25mm


\newif\ifNTTTeX \NTTTeXfalse
\if@english
	\ifx \pfmtname\undefined
		\newskip\kanjiskip
		\newskip\xkanjiskip
	\fi
	\H=0.25mm
	\LoadClass[a4paper]{article}
	\@input{scie.sty}
\else
	\H=1H
	\ifx \jasciikanjiskip\undefined
		\LoadClass{jarticle}
	\else
		\NTTTeXtrue
		\newskip\kanjiskip
		\newskip\xkanjiskip
		\let\kanjiskip=\jintercharskip
		\let\xkanjiskip=\jasciikanjiskip
		\LoadClass{j-article}
	\fi
	\@input{scij.sty}
\fi

\textwidth  172.25mm % 689H = 13Qx53EM
\textheight 248.25mm % 993H = 20Hx50L+13H
\topskip      3.25mm %  13H = \normalsize
\footskip     7.75mm %  32H = \small + 20H
\headheight   2.75mm %  12H = \small
\headsep      5.00mm %  20H
\columnsep    9.75mm %  39H = 3EM
\marginparwidth0mm
\marginparsep0mm

\splittopskip\topskip
\splitmaxdepth\@maxdepth

% margins are based on jarticle.cls
\setlength\topmargin{\paperheight}
\addtolength\topmargin{-\headheight}
\addtolength\topmargin{-\headsep}
\addtolength\topmargin{-\textheight}
\addtolength\topmargin{-\footskip}
\setlength\topmargin{.5\topmargin}
\addtolength\topmargin{-1in}
\@settopoint\topmargin

\setlength\@tempdima{\paperwidth}
\addtolength\@tempdima{-\textwidth}
\setlength\oddsidemargin{.5\@tempdima}
\addtolength\oddsidemargin{-1in}
\evensidemargin\oddsidemargin
\@settopoint\oddsidemargin
\@settopoint\evensidemargin

\def\ryakurekinamesize{\orig@normalsize\@setfontsize\ryakurekinamesize\@xpt{20\H}\@setzw{3.25mm}%
	\parindent1\zw
	\kanjiskip 0mm plus.1\zw minus.05\zw
	\xkanjiskip .2\zw plus.2\zw minus.1\zw
}
\def\ryakurekicharsize{\orig@small\@setfontsize\ryakurekicharsize\@ixpt{18\H}\@setzw{3mm}%
	\parindent1\zw
	\kanjiskip 0mm plus.1\zw minus.05\zw
	\xkanjiskip .2\zw plus.2\zw minus.1\zw
}
\let\rubytinyfont\relax
\def\rubytiny{\@setfontsize\rubytiny\@vpt{11\H}\@setzw{1.75mm}%
	\parindent1\zw
	\kanjiskip 0mm plus.1\zw minus.05\zw
	\xkanjiskip .2\zw plus.2\zw minus.1\zw
}
\def\addresssize{\orig@small\@setfontsize\addresssize\@ixpt{18\H}\@setzw{3mm}%
	\parindent1\zw
	\kanjiskip 0mm plus.1\zw minus.05\zw
	\xkanjiskip .2\zw plus.2\zw minus.1\zw
}

\newif\if@veryshortjournaltitle
\@veryshortjournaltitlefalse
\if@bookrev
	\@veryshortjournaltitletrue
	\RequirePackage{multicol}
\else
	\if@twocol
		%\RequirePackage{twocolumn}
		\twocolumn
	\fi
\fi

\@input{sci209.sty}

% for floatings
\c@topnumber=4
\c@bottomnumber=4
\c@totalnumber=8
\def\topfraction{.95}
\def\dbltopfraction{.95}
\def\bottomfraction{.95}
\def\dblbottomfraction{.95}
\def\textfraction{.01}
\def\floatpagefraction{.7}
\def\dblfloatpagefraction{.7}

% for epsbox.sty
\let\viiipt=\small

\def\enddocument{%
 \if@lastpagebalancing \global\let\@outputdblcol=\balancing@outputdblcol\fi
 \ifnum\c@IHdiffpage=0 \if@footnoteexist\@makerightfootnotetrue\relax
  \else\@makerightfootnotefalse\relax\fi
 \else\@makerightfootnotefalse\relax
 \fi%
 \if@twocol\if@makerightfootnote\let\@makecol=\@makecoloriginal\relax\fi
 \else\let\@makecol=\@makecoloriginal\relax\fi
 \@enddocumenthook
 \@checkend{document}
  \immediate\write\@auxout{% YA
  \string\newlabel{eiko:addresscount}{{\thenumberofaddress}{\thepage}}
  \string\newlabel{eiko:lastpage}{{\thepage}{\thepage}}}%
 %  \@enddocumenthook
 %  \@checkend{document}%
   \clearpage
   \begingroup
     \if@filesw
       \immediate\closeout\@mainaux
       \let\@setckpt\@gobbletwo
       \let\@newl@bel\@testdef
       \@tempswafalse
       \makeatletter \input\jobname.aux
     \fi
     \@dofilelist
     \ifdim \font@submax >\fontsubfuzz\relax
       \@font@warning{Size substitutions with differences\MessageBreak
                  up to \font@submax\space have occured.\@gobbletwo}%
     \fi
     \@defaultsubs
     \@refundefined
     \if@filesw
       \ifx \@multiplelabels \relax
         \if@tempswa
           \@latex@warning@no@line{Label(s) may have changed.
               Rerun to get cross-references right}%
         \fi
       \else
         \@multiplelabels
       \fi
     \fi
   \endgroup
   \deadcycles\z@\@@end}
