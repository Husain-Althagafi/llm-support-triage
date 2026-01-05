LABEL_TO_TRIAGE = {

    # ------------------
    # Card issues
    # ------------------
    "activate_my_card": ("Card Issues", "P2"),
    "card_about_to_expire": ("Card Issues", "P3"),
    "card_acceptance": ("Card Issues", "P2"),
    "card_arrival": ("Card Issues", "P2"),
    "card_delivery_estimate": ("Card Issues", "P3"),
    "card_linking": ("Card Issues", "P2"),
    "card_not_working": ("Card Issues", "P1"),
    "card_payment_not_recognised": ("Card Issues", "P1"),
    "card_payment_fee_charged": ("Card Issues", "P2"),
    "card_payment_wrong_exchange_rate": ("Card Issues", "P2"),
    "card_swallowed": ("Card Issues", "P0"),
    "change_pin": ("Card Issues", "P2"),
    "compromised_card": ("Card Issues", "P0"),
    "contactless_not_working": ("Card Issues", "P2"),
    "declined_card_payment": ("Card Issues", "P1"),
    "disposable_card_limits": ("Card Issues", "P3"),
    "get_disposable_virtual_card": ("Card Issues", "P3"),
    "get_physical_card": ("Card Issues", "P3"),
    "getting_spare_card": ("Card Issues", "P3"),
    "getting_virtual_card": ("Card Issues", "P3"),
    "lost_or_stolen_card": ("Card Issues", "P0"),
    "order_physical_card": ("Card Issues", "P3"),
    "pending_card_payment": ("Card Issues", "P3"),
    "pin_blocked": ("Card Issues", "P1"),
    "virtual_card_not_working": ("Card Issues", "P1"),
    "visa_or_mastercard": ("Card Issues", "P3"),

    # ------------------
    # Payments & transfers
    # ------------------
    "beneficiary_not_allowed": ("Payments & Transfers", "P2"),
    "cancel_transfer": ("Payments & Transfers", "P2"),
    "declined_transfer": ("Payments & Transfers", "P1"),
    "direct_debit_payment_not_recognised": ("Payments & Transfers", "P1"),
    "failed_transfer": ("Payments & Transfers", "P1"),
    "pending_transfer": ("Payments & Transfers", "P3"),
    "receiving_money": ("Payments & Transfers", "P3"),
    "request_refund": ("Payments & Transfers", "P2"),
    "transfer_fee_charged": ("Payments & Transfers", "P2"),
    "transfer_into_account": ("Payments & Transfers", "P3"),
    "transfer_not_received_by_recipient": ("Payments & Transfers", "P1"),
    "transfer_timing": ("Payments & Transfers", "P3"),
    "transaction_charged_twice": ("Payments & Transfers", "P1"),

    # ------------------
    # Cash & ATM
    # ------------------
    "atm_support": ("Cash & ATM", "P3"),
    "cash_withdrawal_charge": ("Cash & ATM", "P2"),
    "cash_withdrawal_not_recognised": ("Cash & ATM", "P1"),
    "declined_cash_withdrawal": ("Cash & ATM", "P1"),
    "pending_cash_withdrawal": ("Cash & ATM", "P3"),
    "wrong_amount_of_cash_received": ("Cash & ATM", "P1"),
    "wrong_exchange_rate_for_cash_withdrawal": ("Cash & ATM", "P2"),

    # ------------------
    # Top ups
    # ------------------
    "automatic_top_up": ("Top Ups", "P3"),
    "balance_not_updated_after_bank_transfer": ("Top Ups", "P2"),
    "balance_not_updated_after_cheque_or_cash_deposit": ("Top Ups", "P2"),
    "pending_top_up": ("Top Ups", "P3"),
    "top_up_by_bank_transfer_charge": ("Top Ups", "P2"),
    "top_up_by_card_charge": ("Top Ups", "P2"),
    "top_up_by_cash_or_cheque": ("Top Ups", "P3"),
    "top_up_failed": ("Top Ups", "P1"),
    "top_up_limits": ("Top Ups", "P3"),
    "top_up_reverted": ("Top Ups", "P2"),
    "topping_up_by_card": ("Top Ups", "P3"),
    "verify_top_up": ("Top Ups", "P2"),

    # ------------------
    # Fees & exchange
    # ------------------
    "exchange_charge": ("Fees & Charges", "P2"),
    "exchange_rate": ("Fees & Charges", "P3"),
    "exchange_via_app": ("Fees & Charges", "P3"),
    "extra_charge_on_statement": ("Fees & Charges", "P2"),
    "Refund_not_showing_up": ("Fees & Charges", "P2"),
    "supported_cards_and_currencies": ("Fees & Charges", "P3"),
    "wrong_exchange_rate_for_cash_withdrawal": ("Fees & Charges", "P2"),

    # ------------------
    # Account & profile
    # ------------------
    "age_limit": ("Account & Profile", "P3"),
    "edit_personal_details": ("Account & Profile", "P3"),
    "passcode_forgotten": ("Account & Profile", "P1"),
    "terminate_account": ("Account & Profile", "P2"),

    # ------------------
    # Identity & verification
    # ------------------
    "unable_to_verify_identity": ("Identity & Verification", "P1"),
    "verify_my_identity": ("Identity & Verification", "P2"),
    "verify_source_of_funds": ("Identity & Verification", "P2"),
    "why_verify_identity": ("Identity & Verification", "P3"),

    # ------------------
    # Phone / device
    # ------------------
    "lost_or_stolen_phone": ("Account & Profile", "P0"),
}


FIRST_STEPS = ["Confirm account/device details (avoid sharing sensitive info)",
        "Clarify when the issue started and any recent actions",
        "Check system status and known issues for this category",
        "Provide next steps or escalate if unresolved"]