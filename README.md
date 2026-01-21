rs_nba_project

Проблема
Сократить цикл принятия решения по региональной кампании и снизить нагрузку на аналитику: система автоматически предлагает план кампании и формирует базы для A/B теста.

Функциональность
- Выбор действия на 11 дней: SMS+скидка 7% (SMS_7), SMS без скидки (SMS_REM), PUSH (PUSH), ничего (NONE)
- Оценка ожидаемой инкрементальной валовой прибыли с учётом затрат (SMS=9 руб., PUSH=0)
- 3 плана: min / rec / max
- Генерация файлов ОС/КГ: outputs/campaigns/<campaign_id>/<plan>/

Демо-артефакты
- data/marts/features_current.csv
- models/nba/region=DEMO/*

Запуск
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python -m src.cli.recommend --region DEMO --target_profit 50000 --dry_run
4) python -m src.cli.recommend --region DEMO --target_profit 50000 --confirm --plan rec

Telegram
export TELEGRAM_BOT_TOKEN="..."
python -m src.bot.bot

Команды
/calc <region> <target_profit_rub>
/status
