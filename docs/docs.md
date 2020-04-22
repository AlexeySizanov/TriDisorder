## Общая информация

Рассматриваем stacked-triangular решётку. Все обмены – AF. Треугольная решётка – в плоскости xy. Эти плоскости моделируем как ромбы чтобы узлы можно было представить как кваратную решётку с диагональными связями.

Предел перколяции такой решётки (site-percolation) [~0.2624](https://arxiv.org/abs/1302.0484) (насколько я понял, треугольные плоскости брали треугольными). Bond-percolation ~0.18602 (по той же ссылке).

То есть предельная концентрация примесей у нас ~0.7376

## Первый порядок через спины

$$
\newcommand{\ch}{{\cal H}}
\newcommand{\a}{\alpha}
\newcommand{\b}{\beta}
\newcommand{\ve}{\varepsilon}
\newcommand{\pd}{\partial}
\newcommand{\ab}{{\alpha\beta}}
\newcommand{\vS}{\vec{S}}
\newcommand{\s}{\vec{S}}
\newcommand{\la}{\lambda}
\newcommand{\vmu}{\vec{\mu}}
\newcommand{\vve}{\vec{\varepsilon}}
\newcommand{\beq}{\begin{eqnarray}}
\newcommand{\eeq}{\end{eqnarray}}
\newcommand{\vphi}{\varphi}
\newcommand{\eps}{\epsilon}
\newcommand{\oy}{\overline{y}}
\nonumber
$$

После того как нашли локальный минимум, руками обеспечивая спиновые плоскости на границах, дальше будем искать деформацию с помощью множителей Лагранжа:
$$
\ch'=\frac12 J_{ab}\s_a\s_b + \vmu_c \la_c \s_c
$$
где $\vmu_c$ – единичные вектора, равные $(0, 0, 1)$ на границах, которые будет твистовать.

Будем работать в угловых переменных
$$
y_{az} = (\theta_a, \, \phi_a)
$$
и считать, что начальное равновесие соотвутствует $y=0$.

Условие равновесия:
$$
\begin{eqnarray}
\pd_{bz} \ch' &=& \left (J_{ab}\s_a + \vmu_b \la_b \right) \pd_z \s_b = 0_{bz}, \\
\pd_{\la_c} \ch' &=& \vmu_c \s_c = 0_c
\end{eqnarray}
$$
Слагаемые Лагранжа обеспечивают компенсацию $\vmu$ - компоненты молекуляоного поля, поэтому:
$$
\la_b = -J_{ab} \s_a \vmu_b  \label{eq:la}
$$
Получаем условие равновесия без $\la$:
$$
J_{ab} \left(\delta^\ab - \mu^\a_b \mu^\b_b \right)^\ab_b S^\a_a \pd_z S^\b_b = 0_{bz}
\label{eq:eq}
$$


Наложение твиста равносильно преобразованию:
$$
\begin{eqnarray}
\vmu_b &\rightarrow& \vmu_b + \vec{e}^\eta_b \ve^\eta, \\
y_{az} &\rightarrow& y_{az} + y_{az}^\eta \ve^\eta + 
											\tfrac12 y_{az}^{\eta\rho} \ve^\eta \ve^\rho
\end{eqnarray}
$$
где $\vec{e}_\eta$ – вектор в напрвлении $\eta = (x, y)$ с единичной длиной на твист-границах и нулевой в других местах,   $\ve^\eta$ - соответсвующий инфинитезимальный угол поворота. Раскладывая $\eqref{eq:eq}$ до первого порядка по $\ve$, получаем:
$$
J_{ab} \left( \delta^\ab - \mu^\a_b \mu^\b_b \right)^\ab_b
\left( 
		\pd_{z'} S^\a_a \pd_z S^\b_b y^\eta_{az'} + 
		S^\a_a \pd_{zz'} S^\b_b y^\eta_{bz'}
\right) = 

J_{ab} \left( \mu^\a_b e^{\b\eta}_b + \mu^\b_b e^{\a\eta}_b \right) S^\a_a \pd_z S^\b_b
$$
или
$$
\begin{eqnarray}

P^\ab_b
\left( 
		J_{ab} \pd_{z'} S^\a_a \pd_z S^\b_b + 
	  m^\a_b \pd_{zz'} S^\b_b \delta_{ab}
\right)_{bzaz'} y^\eta_{az'} &=& 

\left[
\left( \mu^\a_b e^{\b\eta}_b + \mu^\b_b e^{\a\eta}_b \right) m^\a_b \pd_z S^\b_b 
\right]^\eta_{bz} 
\end{eqnarray}
$$
где
$$
\begin{eqnarray}
P^\ab_b &=& \delta^\ab - \mu^\a_b \mu^\b_b, \\
\vec{m}_b &=& J_{ab} \s_a
\end{eqnarray}
$$

Эта система уравнений недоопределена и решается методом наименьших квадратов (ну или псевдообратной матрицей, но она плотная, поэтому её искать дорого).

Производные энергии по $\ve$:
$$
\begin{eqnarray}

\pd_\eta \ch &=& \left( \vec{m}_b \, \pd_z \s_b \right) y^\eta_{bz} \label{eq:order1} \\

\pd_{\eta\rho} \ch &=& \left( 
		\vec{m}_b \pd_{zz'} \s_b \delta_{ab} + 
		J_{ab} \pd_z \s_a \pd_{z'} \s_b 
\right) y^\eta_{az} y^\rho_{bz'} +

\left( \vec{m}_b \pd_z \s_b \right) y^{\eta\rho}_{bz} \label{eq:order2}

\end{eqnarray}
$$
Так как конструкция $\vec{m}_a \pd_z \s_b$ отлична от нуля только на твист-гриницах, певая производная определяется только этими границами, а вклад во вторую производную (последнее слагаемое) порядка $1/L$ относительно остальной суммы. И второе слагаемое во второй производной будем игнорировать.







