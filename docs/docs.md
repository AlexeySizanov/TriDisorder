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
\newcommand{\hP}{\hat{P}}
\newcommand{\hR}{\hat{R}}
\nonumber
$$
## Общая информация

Рассматриваем stacked-triangular решётку. Все обмены – AF. Треугольная решётка – в плоскости xy. Эти плоскости моделируем как ромбы, чтобы узлы можно было представить как кваратную решётку с диагональными связями. Примеси – в узлах.

Предел перколяции такой решётки (site-percolation) ~0.2624 [[arXiv]](https://arxiv.org/abs/1302.0484) (насколько я понял, треугольные плоскости там были именно треугольными, не ромбическими). Bond-percolation ~0.18602 (по той же ссылке).

То есть предельная концентрация примесей у нас ~0.7376

## Уравнение равновесия.



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

##### Доводка точки равновесия.

В результате итеративного процесса мы оказваемся рядом с равновесием, но не точно в нём. Раскладывая $\eqref{eq:eq}$ по $y$ до первого порядка, получаем на смещение к точке равновесия:
$$
P^\ab_b
\left( 
		J_{ab} \pd_{z'} S^\a_a \pd_z S^\b_b + 
	  m^\a_b \pd_{zz'} S^\b_b \delta_{ab}
\right)_{bzaz'} y^\eta_{az'} = 

-P^\ab_b m^\a_b \pd_z S^\b_b
$$
где 
$$
\begin{eqnarray}
P^\ab_b &=& \delta^\ab - \mu^\a_b \mu^\b_b, \\
\vec{m}_b &=& J_{ab} \s_a
\end{eqnarray}
$$

##### Вычисление деформации.

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
\right]^\eta_{bz} \label{eq:main}
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
Так как конструкция $\left(\vec{m}_a \pd_z \s_a\right)_{az}$ отлична от нуля только на твист-гриницах, певая производная определяется только этими границами, а вклад во вторую производную (последнее слагаемое) порядка $1/L$ относительно остальной суммы. Поэтому второе слагаемое во второй производной будем игнорировать.

Сама жёсткость при этом получается как собственные значения матрицы
$$
\tfrac1L \pd_{\eta\rho} \ch
$$
Где $L$ –линейный размер решётки. Эти значения получаются не одинаковыми как раз из-за отсутсвия равновесия на границах. Но, так как эффект границ $\sim 1/L$, эти значения должны сближаться с ростом $L$.

## Вычисление через модифицированный гамильтониан

Модификация только уравнения равновесия использовать не получается из-за отсутсвия истиннного равновесия на границах, из-за чего первая производная энергии по деормации не 0. Поэтому предлагается модифицировать сам гамитльтониан условия на границах обеспечивались автоматически:
$$
\ch = \frac12 J_{ab} \;\hat{P}_a \s_a  \cdot \hat{P}_b\s_b
$$
Уравнение равновесия:
$$
J_{ab} \; \hat{P}_a \s_a \cdot \hat{P}_b \pd_z \s_b = 0_{bz}
$$
Раскладывая до первого порядка по $\ve$:
$$
\left(
		J_{ab} \pd_{z'} \s'_a \pd_z \s'_b + 
	  \vec{m}'_b \pd_{zz'} \s'_b \delta_{ab}
\right)_{bz,az'} y^\eta_{az'}

=

- \left(
		J_{ab} \; \hR^\eta_a \s_a \cdot \pd_z \s'_b + 
		\vec{m}'_b \cdot \hR^\eta_b \pd_z \s_b
\right)^\eta_{bz}
$$
где
$$
\s'_a = \hat{P}_a \s_a, \\
\vec{m}'_a = J_{ab} \s'_b ,\\
R^\eta_a = \pd \hat{P}_a / \pd \ve_\eta =- e^\b_{\eta,a} \mu^\a_a - e^a_{\eta,a} \mu^\b_a
$$
Производные энергии:
$$
\begin{eqnarray}
\frac{d \ch}{ d\ve^\eta} &=& \pd_\eta \ch + \pd_{az}\ch\, y^\eta_{ab} 
= 
\pd_\eta \ch
=
J_{ab} \hP_a \s_a \cdot \hR^\eta_b \s_b\\

\frac{d^2 \ch}{ d\ve^\eta d\ve^\rho} &=& \frac{d}{d\ve_\rho} \pd_\eta \ch = \pd_{\eta\rho} \ch + \pd_{\eta,az} \ch y^\rho_{az} = \\

&=& 

J_{ab} \left( 
	\hR^\eta_a \s_a \cdot \hR^\rho_b \s_b + 
	\hP_a \s_a \cdot \hat{T}^{\eta\rho}_b \s_b
\right) + \\

&&
J_{ab} \left(
		\hR^\eta_a \s_a \cdot \hP_b \pd_z \s_b + \hP_a \s_a \cdot \hR^\eta_b \pd_z \s_b
\right) y^\rho_{bz}
\end{eqnarray}
$$
где
$$
\hat{T}^{\eta\rho, \ab}_a = -e^\a_{\eta,a} e^\b_{\rho,a} - e^\b_{\eta,a} e^\a_{\rho,a}
$$


## Технические моменты

Начальное состояние создаётся так:

1. создаётся идеальная 120° структура в плоскости xy ($\theta = 0$).
2. $\theta_a \rightarrow \theta_a + \delta \theta_a, \quad \phi_a \rightarrow \phi_a + \delta \phi_a$,  где $\delta\theta_a,\; \delta\phi_a$ равномерно распределены в [-0.2, 0.2].

Далее происходит оптимизация (поиск локального минимума). Можно делать её  различными модификациями градиентного спуска в пространстве угловых переменных, но на практике это получается очень долго (видимо, из-за большой изрезанности "ландшафта" энергии). Поэтому последовательно применялось следующее преобразование:
$$
\s_a \rightarrow \frac{\b\s_a - (1 - \b) \hat{P}_a \vec{m}_b}{\left| \b \s_a - (1-\b)\hat{P}_a \vec{m}_b \right|}
$$
Такая итеративная процедура (она, конечно, похожа на градиентны спуск) сходится гораздо быстрее. Остановка происходит по правилу 
$$
\max_a\angle (\s_a, -\vec{m}_a) < 0.001^\circ 
$$
*Однако в некоторых случаях алгоритм не сходился за пороговое число итераций, равное 300_000  (какие конигурации в этому приводили я не смотрел)*. Такие конфигурации просто отбрасывались. Возможно, здесь был внесён некоторый bias.

Так же я не проверял во всех случаях корректность решения уравнения $\eqref{eq:main}$.

## Результаты

Для анализа критического поведения будем использовать простой конечно-размерный скейлинг в виде
$$
\rho = L^{\a/\nu} f\left(L^{1/\nu} (c-c_0)\right)
$$
