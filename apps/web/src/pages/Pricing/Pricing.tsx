import { Flex, Text, ScrollArea } from "@radix-ui/themes";
import "./Pricing.css";
import Header from "../../components/Header/Header";
import Calculator from "../../components/PriceCalculator/Calculator";
import PricingCard from "../../components/PricingCard/PricingCard";
import Footer from "../../components/Footer/Footer";
import pricingImageWebp from "../../assets/pricing/pricing-image.webp";
import pricingImageJpg from "../../assets/pricing/pricing-image-85.jpg";

export default function Pricing() {
  return (
    <Flex
      direction="column"
      style={{
        position: "relative",
        height: "100%",
        width: "100%",
      }}
      className="pulsing-background"
    >
      <ScrollArea type="scroll">
        <div>
          <div className="pricing-main-container">
            <div className="pricing-image-container">
              <picture>
                <source srcSet={pricingImageWebp} type="image/webp" />
                <img
                  src={pricingImageJpg}
                  alt="pricing hero"
                  className="pricing-hero-image fade-in"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
              </picture>
              <div className="pricing-gradient-overlay"></div>
            </div>
            <div className="pricing-content-container">
              <Flex className="header-container">
                <div style={{ maxWidth: "1312px", width: "100%" }}>
                  <Header px="0px" />
                </div>
              </Flex>
              <Flex className="pricing-hero-container">
                <Flex className="text-container-pricing" direction="column">
                  <Text
                    size="9"
                    weight="bold"
                    trim="both"
                    className="pricing-title"
                    mb="24px"
                  >
                    Pricing
                  </Text>
                  <Text
                    className="white pricing-description"
                    size="5"
                    weight="medium"
                    style={{
                      maxWidth: "542px",
                      lineHeight: "32px",
                    }}
                  >
                    Flexible plans for every stage of your journey. From solo
                    developers to large-scale enterprises.
                  </Text>
                </Flex>
              </Flex>
              <Flex className="pricing-cards-container">
                <Flex
                  direction="column"
                  gap="8"
                  style={{ flex: 1, width: "100%" }}
                >
                  <Calculator />
                </Flex>
                <Flex direction="column" style={{ flex: 1, width: "100%" }}>
                  <Flex direction="column" gap="52px">
                    <PricingCard
                      icon={
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <rect
                            width="24"
                            height="24"
                            fill="white"
                            fillOpacity="0.01"
                          />
                          <path
                            fillRule="evenodd"
                            clipRule="evenodd"
                            d="M10.9658 6.16568L12.2457 4.88565C13.277 3.85441 15.0858 3.13875 16.8833 2.882C17.769 2.75547 18.6066 2.7476 19.2903 2.84998C19.998 2.95597 20.4317 3.16313 20.6345 3.36571C20.837 3.56829 21.0442 4.002 21.1501 4.70971C21.2525 5.39337 21.2447 6.23102 21.1181 7.11681C20.8613 8.91422 20.1457 10.723 19.1143 11.7543L12 18.8686L10.1657 17.0344C9.85328 16.7219 9.34675 16.7219 9.03434 17.0344C8.72192 17.3467 8.72192 17.8533 9.03434 18.1658L11.4344 20.5658C11.7468 20.8781 12.2533 20.8781 12.5657 20.5658L13.4412 19.6902L15.314 22.8115C15.4408 23.0229 15.6578 23.1638 15.9023 23.1941C16.1468 23.2242 16.3916 23.1398 16.5657 22.9658L19.7657 19.7658C19.9437 19.5877 20.0276 19.3362 19.9921 19.0869L19.2485 13.8829L20.2458 12.8857C21.6145 11.5169 22.4188 9.32568 22.702 7.34309C22.8455 6.33888 22.8626 5.34155 22.7325 4.47273C22.606 3.62797 22.3231 2.79171 21.7657 2.23432C21.2084 1.67694 20.3721 1.39413 19.5273 1.26762C18.6585 1.13751 17.6612 1.15463 16.6568 1.29808C14.6743 1.58129 12.4831 2.3856 11.1144 3.75427L10.1172 4.75149L4.91319 4.00806C4.66392 3.97245 4.41242 4.05627 4.23437 4.23433L1.03437 7.43432C0.860175 7.60853 0.77595 7.85326 0.806049 8.09776C0.83615 8.34225 0.977217 8.55926 1.18846 8.68601L4.30987 10.5588L3.43439 11.4343C3.28437 11.5843 3.20008 11.7878 3.20008 12C3.20008 12.2122 3.28437 12.4157 3.43439 12.5657L5.8344 14.9657C6.14682 15.2781 6.65335 15.2781 6.96576 14.9657C7.27819 14.6533 7.27819 14.1467 6.96576 13.8343L5.13144 12L6.16576 10.9657L10.9658 6.16568ZM16.1589 21.1098L14.6074 18.524L17.8343 15.2971L18.3516 18.9171L16.1589 21.1098ZM5.47607 9.39265L8.70301 6.16569L5.0829 5.64854L2.89026 7.84118L5.47607 9.39265ZM3.76575 16.5656C4.07816 16.2533 4.07816 15.7467 3.76575 15.4343C3.45333 15.1219 2.94679 15.1219 2.63437 15.4343L1.03437 17.0342C0.72195 17.3467 0.72195 17.8533 1.03437 18.1656C1.34679 18.4781 1.85333 18.4781 2.16574 18.1656L3.76575 16.5656ZM6.16578 18.9658C6.47819 18.6533 6.47821 18.1467 6.16579 17.8344C5.85338 17.5219 5.34685 17.5219 5.03442 17.8344L1.83437 21.0342C1.52195 21.3467 1.52195 21.8533 1.83437 22.1656C2.14679 22.4781 2.65331 22.4781 2.96573 22.1658L6.16578 18.9658ZM8.56575 21.3656C8.87816 21.0533 8.87816 20.5467 8.56575 20.2342C8.25333 19.9219 7.74679 19.9219 7.43437 20.2342L5.83437 21.8342C5.52195 22.1467 5.52195 22.6533 5.83437 22.9656C6.14679 23.2781 6.65333 23.2781 6.96575 22.9656L8.56575 21.3656ZM15.2 10.7981C16.3036 10.7981 17.1981 9.90352 17.1981 8.8C17.1981 7.69648 16.3036 6.80189 15.2 6.80189C14.0965 6.80189 13.2019 7.69648 13.2019 8.8C13.2019 9.90352 14.0965 10.7981 15.2 10.7981Z"
                            fill="white"
                          />
                        </svg>
                      }
                      title="High Volume"
                      subtitle="Scalable & high-throughput solution"
                      checkpoints={[
                        "Discounted per-page rate",
                        "Integrate in minutes",
                        "99.99% SLA",
                        "No Data retention",
                        "Custom ingestion pipeline",
                        "0 limits and high throughput",
                        "24/7 founder support",
                      ]}
                    />
                    <PricingCard
                      icon={
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <rect
                            width="24"
                            height="24"
                            fill="white"
                            fillOpacity="0.01"
                          />
                          <path
                            fillRule="evenodd"
                            clipRule="evenodd"
                            d="M11.6618 1.27505C11.8762 1.17498 12.124 1.17498 12.3384 1.27505L21.9383 5.75505C22.2201 5.88649 22.4001 6.16918 22.4001 6.48V17.52C22.4001 17.8309 22.2201 18.1134 21.9383 18.245L12.3384 22.725C12.124 22.825 11.8762 22.825 11.6618 22.725L2.06179 18.245C1.78013 18.1134 1.6001 17.8309 1.6001 17.52V6.48C1.6001 6.16918 1.78013 5.88649 2.06179 5.75505L11.6618 1.27505ZM3.2001 7.68924L11.2001 11.0892V20.7438L3.2001 17.0106V7.68924ZM12.8001 20.7438L20.8001 17.0106V7.68924L12.8001 11.0892V20.7438ZM12.0001 9.69075L19.6351 6.44585L12.0001 2.88281L4.36504 6.44585L12.0001 9.69075Z"
                            fill="white"
                          />
                        </svg>
                      }
                      title="Self-hosted"
                      subtitle="Maximum control, security, & customization"
                      checkpoints={[
                        "Flat monthly rate",
                        "White glove onboarding",
                        "Self-hosted in private VPC",
                        "Deploy on AWS/GCP/Azure",
                        "Custom ingestion pipeline",
                        "0 limits and high throughput",
                        "24/7 founder support",
                      ]}
                    />
                  </Flex>
                </Flex>
              </Flex>
            </div>
          </div>
        </div>
        <Footer />
      </ScrollArea>
    </Flex>
  );
}
