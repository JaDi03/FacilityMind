import time
import logging
import uuid

logger = logging.getLogger(__name__)

# ==============================================================================
# Circle Gateway Nanopayments (x402 Protocol)
# ==============================================================================
# Reference: https://developers.circle.com/gateway/nanopayments#how-it-works
# 
# The x402 protocol is based on HTTP 402 Payment Required.
# 1. Buyer requests a paid resource.
# 2. Seller responds with 402 Payment Required and payment details.
# 3. Buyer signs an EIP-3009 payment authorization (offchain, zero gas).
# 4. Buyer retries request with signed authorization attached.
# 5. Seller verifies the signature and serves the resource immediately.
# 6. Gateway collects authorizations and settles them in batches onchain.

# Gemini 2.5 Pro Base Pricing
INPUT_TOKEN_PRICE = 0.00000125  # $1.25 per 1M tokens
OUTPUT_TOKEN_PRICE = 0.000005   # $5.00 per 1M tokens
MARGIN_MULTIPLIER = 2.0         # 100% markup

class X402NanopaymentGateway:
    """
    Middleware que implementa el protocolo x402 de Circle Gateway.
    No ejecuta transferencias en cadena (onchain) individuales, sino que valida
    las firmas EIP-3009 offchain para un "Batched Settlement" posterior.
    """
    
    @staticmethod
    def generate_402_invoice(input_chars: int, output_chars: int) -> dict:
        """Generates the invoice required for the x402 protocol."""
        input_tokens = int(input_chars / 4)
        output_tokens = int(output_chars / 4)
        
        raw_cost = (input_tokens * INPUT_TOKEN_PRICE) + (output_tokens * OUTPUT_TOKEN_PRICE)
        final_charge = max(raw_cost * MARGIN_MULTIPLIER, 0.005) # Minimo medio centavo
        
        return {
            "status_code": 402,
            "error": "Payment Required",
            "amount_usdc": round(final_charge, 6),
            "payment_address": "0xGatewayWalletContract",
            "challenge": str(uuid.uuid4())
        }

    @staticmethod
    def verify_eip3009_authorization(invoice: dict, signature: str = "mock_eip3009_signature") -> dict:
        """
        Verifica la firma offchain EIP-3009 (Zero Gas).
        En milisegundos y sin contacto con la blockchain, libera el recurso.
        """
        # Simulacion de validacion criptografica offchain (milisegundos)
        time.sleep(0.01)
        
        # El Gateway recolecta estas autorizaciones en memoria para liquidarlas 
        # en lote (Batched Settlement) mas tarde.
        receipt = {
            "network": "Circle Gateway (x402)",
            "currency": "USDC",
            "amount_charged": invoice["amount_usdc"],
            "eip3009_auth": signature[:15] + "...",
            "settlement_status": "PENDING_BATCH_SETTLEMENT",
            "input_tokens": int(invoice["amount_usdc"] / MARGIN_MULTIPLIER / INPUT_TOKEN_PRICE), # rough estimation for receipt
            "output_tokens": 0 
        }
        
        logger.info(f"[x402 Protocol] EIP-3009 Auth valid. {receipt['amount_charged']} USDC added to batch.")
        return receipt

    @staticmethod
    def process_payment(input_chars: int, output_chars: int) -> dict:
        """
        Wrapper que simula el ciclo completo x402 para la interfaz de FastAPI/Streamlit:
        Genera el invoice (402) -> 'Recibe' la firma -> Verifica offchain -> Libera recurso.
        """
        invoice = X402NanopaymentGateway.generate_402_invoice(input_chars, output_chars)
        
        # En una arquitectura pura, aqui regresariamos un HTTP 402 al frontend.
        # Como estamos procesando del lado del backend para la automatizacion, 
        # asumimos la firma instantanea por la wallet programable.
        
        receipt = X402NanopaymentGateway.verify_eip3009_authorization(invoice)
        receipt["input_tokens"] = int(input_chars / 4)
        receipt["output_tokens"] = int(output_chars / 4)
        receipt["tx_hash"] = "x402_offchain_" + str(uuid.uuid4().hex)[:8]
        return receipt
